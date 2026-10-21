import argparse
import fcntl
import hashlib
import time
import platform
from functools import lru_cache
import os
import shutil
import signal
import subprocess
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from threading import Lock, BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from reproduce.install_Hydrangea_environments import read_runner_environment
from run_repair_single import normalize_patch, MODEL_SETTING_LINE, LOGGING_LINE, OUTPUT_REPLACEMENT_LINE, BEHAVIORAL_ISSUE
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

from paper_results import DATASETS

METHODS = ["SWE-agent", "Agentless", "AutoCodeRover", "SWE-agent*", "Agentless*", "LocAgent", "Codex", "LIDL"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-5.6-terra"
EVALUATION_VERSION = 6
DOCKER_SLOTS = BoundedSemaphore(4)
ENVIRONMENT_REASONS = {"image_architecture_mismatch", "image_unavailable", "test_dependency_missing", "baseline_test_failed", "harness_path_error", "execution_interrupted", "setup_timeout", "setup_failed", "nltk_resource_missing", "harness_annotation_error", "mock_fallback_used", "test_import_failed"}
IMAGE_OVERRIDES = {}
PATCH_BLOCK = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DIFF_FILE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
HUNK = re.compile(r"^@@ ", re.MULTILINE)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def generation_path(directory):
    trace = directory / "trace.json"
    return trace if trace.is_file() else directory / "output.json"


def localization_record(directory):
    output = read_json(generation_path(directory))
    return output["localization"] if "localization" in output else read_json(directory / "localization.json")


def evaluation_record(directory):
    trace = directory / "trace.json"
    if trace.is_file():
        return read_json(trace).get("evaluation", {"status": "not_evaluated"})
    path = directory / "patch_result.json"
    return read_json(path) if path.is_file() else {"status": "not_evaluated"}


def save_evaluation(directory, result):
    trace = directory / "trace.json"
    if trace.is_file():
        record = read_json(trace)
        previous = record.get("evaluation")
        if previous and previous != result:
            record.setdefault("evaluation_history", []).append(previous)
        record["evaluation"] = result
        write_json(trace, record)
    else:
        write_json(directory / "patch_result.json", result)


def discover_instances(paths, repair_model, localization_model):
    roots = paths or [PROJECT_ROOT / "results/results_repair"]
    instances = []
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file() and root.name in {"output.json", "patch_result.json", "trace.json"}:
            instances.append(root.parent)
        elif generation_path(root).is_file():
            instances.append(root)
        elif root.is_dir():
            instances.extend(path.parent for name in ("output.json", "trace.json") for path in root.rglob(name))
        else:
            logger.warning(f"Repair result path does not exist: {root}")
    selected = []
    for result_dir in sorted(set(instances)):
        localization_path = result_dir / "localization.json"
        if not localization_path.is_file() and not (result_dir / "trace.json").is_file():
            logger.warning(f"Missing localization.json in {result_dir}")
            continue
        localization = localization_record(result_dir)
        if result_dir.parents[2].name != repair_model:
            continue
        if localization.get("model") != localization_model:
            continue
        selected.append(result_dir)
    return selected


def patch_statistics(patch):
    match = PATCH_BLOCK.fullmatch(patch.strip())
    if match:
        patch = match.group(1)
    changed_lines = [
        line for line in patch.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    return {
        "changed_files": len({match.group(2) for match in DIFF_FILE.finditer(patch)}),
        "hunks": len(HUNK.findall(patch)),
        "added_lines": sum(line.startswith("+") for line in changed_lines),
        "deleted_lines": sum(line.startswith("-") for line in changed_lines),
    }


def first_usage(trace_path):
    if not trace_path.is_file():
        return None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        usage = (((record.get("data") or {}).get("output") or {}).get("usage"))
        if usage is not None:
            return usage
    return None


def instance_row(result_dir):
    result_path = result_dir / "patch_result.json"
    result = evaluation_record(result_dir)
    localization = localization_record(result_dir)
    repair_output = read_json(generation_path(result_dir))
    generation = repair_output.get("generation", repair_output)
    attempts = generation.get("attempts") or []
    patch_path = result_dir / "patch.diff"
    patch = (
        str((attempts[0].get("response") or {}).get("patch") or "")
        if attempts
        else patch_path.read_text(encoding="utf-8") if patch_path.is_file() else ""
    )
    usage = next((entry.get("data", {}).get("output", {}).get("usage") for entry in repair_output.get("api_records", []) if entry.get("data", {}).get("output", {}).get("usage")), None) or first_usage(result_dir / "trace.jsonl") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    status = result.get("status")
    reason = failure_reason(result, repair_output.get("status"))
    if reason in ENVIRONMENT_REASONS:
        status = "environment_error"
    evaluated_statuses = {"resolved", "unresolved", "missing_patch", "invalid_patch"}
    if status not in evaluated_statuses | {"not_evaluated", "error", "environment_error"}:
        raise ValueError(f"Unknown patch evaluation status for {result_dir}: {status!r}")
    if attempts and len(attempts) != 1:
        logger.warning(f"Multiple repair attempts cannot establish Pass@1 for {result_dir}")
        status = "not_evaluated"
    if result_path.is_file() and patch_path.is_file() and patch_path.stat().st_mtime_ns > result_path.stat().st_mtime_ns:
        logger.warning(f"Patch is newer than its test result: {result_dir}")
        status = "not_evaluated"
    if status == "resolved":
        execution = (result.get("test") or {}).get("execution") or {}
        if execution.get("returncode") != 0 or execution.get("timed_out"):
            logger.warning(f"Resolved result lacks successful test execution: {result_dir}")
            status = "not_evaluated"
    pass_at_1 = 1 if status == "resolved" else 0 if status in evaluated_statuses else None
    return {
        "dataset": localization["dataset"],
        "repair_model": result_dir.parents[2].name,
        "approach": localization["approach"],
        "localization_model": localization["model"],
        "instance_id": localization["instance_id"],
        "status": status,
        "failure_reason": reason,
        "pass_at_1": pass_at_1,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        **patch_statistics(patch),
        "result_path": str(result_path),
    }


def average(rows, field):
    values = [row[field] for row in rows if isinstance(row[field], (int, float))]
    return sum(values) / len(values) if values else None


def summarize(rows, scope, scope_value):
    not_evaluated = sum(row["pass_at_1"] is None for row in rows)
    resolved = sum(row["pass_at_1"] == 1 for row in rows)
    total = len(rows)
    return {
        "scope": scope,
        "scope_value": scope_value,
        "instances": total,
        "evaluated_instances": total - not_evaluated,
        "not_evaluated_instances": not_evaluated,
        "resolved": resolved,
        "pass_at_1": resolved / total if total and not_evaluated == 0 else None,
        "avg_input_tokens": average(rows, "input_tokens"),
        "avg_output_tokens": average(rows, "output_tokens"),
        "avg_total_tokens": average(rows, "total_tokens"),
        "avg_changed_files": average(rows, "changed_files"),
        "avg_hunks": average(rows, "hunks"),
        "avg_changed_lines": average(
            [{**row, "changed_lines": row["added_lines"] + row["deleted_lines"]} for row in rows],
            "changed_lines",
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def collect_benchmark_rows(paths, repair_model, localization_model, limit):
    discovered = discover_instances(paths, repair_model, localization_model)
    by_key = {}
    for path in discovered:
        metadata = localization_record(path)
        key = (metadata["dataset"], metadata["approach"], metadata["instance_id"])
        if key in by_key:
            raise ValueError(f"Multiple repair runs for {key}; select one explicit result root")
        by_key[key] = path
    rows = []
    for dataset in DATASETS:
        instances = read_json(dataset["dataset_json"])
        if limit is not None:
            logger.warning(f"Validation uses the first {limit} real instances of {dataset['name']}")
            instances = instances[:limit]
        for method in METHODS:
            for instance in instances:
                key = (dataset["name"], method, instance["instance_id"])
                path = by_key.get(key)
                if path is not None:
                    row = instance_row(path)
                else:
                    row = {
                        "dataset": key[0], "repair_model": repair_model, "approach": method,
                        "localization_model": localization_model, "instance_id": key[2],
                        "status": "not_evaluated", "failure_reason": None, "pass_at_1": None,
                        "input_tokens": None, "output_tokens": None, "total_tokens": None,
                        **patch_statistics(""), "result_path": "",
                    }
                rows.append(row)
    return rows


def write_repair_table(output_dir, summaries):
    methods = {row["scope_value"]: row for row in summaries if row["scope"] == "approach"}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Downstream automated repair using context from different localization methods. Pass@1 uses the full selected benchmark denominator. Resolved is the number of patches passing the recorded test procedure. TBD indicates incomplete evaluation.}",
             r"\label{tab:repair_comparison}", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{lcc}", r"\toprule", r"Method & Pass@1 (\%) & Resolved \\", r"\midrule"]
    for method in METHODS:
        row = methods[method]
        pending = row["not_evaluated_instances"] > 0
        value = "TBD" if pending else f"{100 * row['pass_at_1']:.2f}"
        resolved = "TBD" if pending else str(row["resolved"])
        if method == "LIDL":
            lines.append(r"\midrule")
        name = method.replace("*", "$^*$")
        lines.append(f"{name} & {value} & {resolved}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output_dir / "repair_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_patch(patch):
    match = PATCH_BLOCK.fullmatch(patch.strip())
    normalized = match.group(1) if match else patch
    if not normalized.startswith("diff --git "):
        raise ValueError("Patch is not a git unified diff")
    normalized = normalized.rstrip("\n") + "\n"
    lines = normalized.splitlines(keepends=True)
    header = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*?)(?:\n)?$")
    changes = []
    for index, line in enumerate(lines):
        match = header.match(line)
        if not match:
            continue
        old_count = new_count = 0
        for body in lines[index + 1:]:
            if body.startswith(("@@ ", "diff --git ")):
                break
            if body.startswith("\\ No newline at end of file"):
                continue
            if not body.startswith((" ", "+", "-")):
                raise ValueError(f"Invalid diff body after hunk at line {index + 1}")
            old_count += body.startswith((" ", "-"))
            new_count += body.startswith((" ", "+"))
        replacement = f"@@ -{match[1]},{old_count} +{match[2]},{new_count} @@{match[3]}\n"
        if replacement != line:
            changes.append({"line": index + 1, "before": line, "after": replacement})
            lines[index] = replacement
    result = "".join(lines)
    return result, {"policy": "recount_hunk_lengths_only", "header_changes": changes,
                    "raw_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                    "normalized_sha256": hashlib.sha256(result.encode()).hexdigest(),
                    "normalized_patch": result if result != patch else None}


def copy_isolated_tree(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(["cp", "-a", "--reflink=auto", str(source) + "/.", str(destination)], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Isolated copy failed: {result.stderr}")


def check_patch(repo_path, patch_text):
    patch_paths = {
        match.group(1)
        for match in re.finditer(r"^(?:--- a/|\+\+\+ b/)(.+)$", patch_text, re.MULTILINE)
        if match.group(1) != "/dev/null"
    }
    invalid_paths = [
        file_path
        for file_path in patch_paths
        if Path(file_path).is_absolute() or ".." in Path(file_path).parts
    ]
    if invalid_paths:
        return {
            "valid": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"Patch contains unsafe paths: {sorted(invalid_paths)}",
            "syntax_errors": [],
        }
    with tempfile.TemporaryDirectory(prefix="lidl-patch-check-") as temp_dir:
        temp_repo = Path(temp_dir) / "repo"
        temp_repo.mkdir()
        for file_path in patch_paths:
            source_path = repo_path / file_path
            if not source_path.resolve().is_relative_to(repo_path.resolve()):
                raise ValueError(f"Patch path escapes repository through symlink: {file_path}")
            if source_path.is_file():
                target_path = temp_repo / file_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
        result = subprocess.run(
            ["git", "apply", "-"],
            cwd=temp_repo,
            input=patch_text,
            text=True,
            capture_output=True,
            check=False,
        )
        syntax_errors = []
        if result.returncode == 0:
            changed_python_files = sorted(
                {
                    match.group(1)
                    for match in re.finditer(r"^\+\+\+ b/(.+\.py)$", patch_text, re.MULTILINE)
                    if (temp_repo / match.group(1)).is_file()
                }
            )
            for file_path in changed_python_files:
                compile_result = subprocess.run(
                    [sys.executable, "-m", "py_compile", file_path],
                    cwd=temp_repo,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if compile_result.returncode != 0:
                    syntax_errors.append({"file": file_path, "stdout": compile_result.stdout, "stderr": compile_result.stderr})
        return {
            "valid": result.returncode == 0 and not syntax_errors,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "syntax_errors": syntax_errors,
        }


def validate_patch(instance, repo_path, patch_text):
    validation = check_patch(repo_path, patch_text)
    errors = []
    if validation["returncode"] != 0:
        errors.append(validation["stderr"] or "git apply failed")
    errors.extend(item["stderr"] or item["stdout"] for item in validation["syntax_errors"])
    changed_lines = [
        line for line in patch_text.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    if not changed_lines:
        errors.append("The patch contains no source line changes.")
    behavioral_issue = BEHAVIORAL_ISSUE.search(instance["problem_statement"])
    if behavioral_issue and changed_lines and all(MODEL_SETTING_LINE.match(line) for line in changed_lines):
        errors.append(
            "The behavioral defect patch only changes model settings; repair the prompt, "
            "message construction, logic, or data flow instead."
        )
    if behavioral_issue and changed_lines and all(LOGGING_LINE.match(line) for line in changed_lines):
        errors.append("Changing only logging cannot repair the reported runtime behavior.")
    # A string replacement alone does not prove an invalid repair; tests assess its behavior.
    return {**validation, "errors": errors, "valid": not errors}


FIXED_PATCHED_REPO = re.compile(
    r'^PATCHED_REPO = Path\("/tmp"\) / f"lidl-\{INSTANCE_ID\}-patched"$',
    re.MULTILINE,
)
AGENTISSUE_SUCCESS = (
    "PATCH SUCCEEDED",
    "PATCH SUCCESSFULLY VERIFIED",
    "FIX SUCCESSFULLY VERIFIED",
    "NO BUG",
    "FIX CONFIRMED",
    "PATCH VERIFIED",
    "patched succeeded",
)
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")
HYDRANGEA_INSTANCE_LOCKS = {}
HYDRANGEA_LOCKS_GUARD = Lock()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_patch(result_dir, repair_output):
    generation = repair_output.get("generation", repair_output)
    attempts = generation.get("attempts") or []
    if not attempts:
        patch_path = result_dir / "patch.diff"
        return patch_path.read_text(encoding="utf-8") if patch_path.is_file() else None
    patch = str((attempts[0].get("response") or {}).get("patch") or "").strip()
    match = PATCH_BLOCK.fullmatch(patch)
    if match:
        patch = match.group(1).strip()
    return f"{patch}\n" if patch else None


def safe_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.lower() for marker in SECRET_MARKERS)
    }


def hydrangea_instance_lock(instance_id):
    with HYDRANGEA_LOCKS_GUARD:
        return HYDRANGEA_INSTANCE_LOCKS.setdefault(instance_id, Lock())


def copy_hydrangea_harness(instance_id, patch, root):
    source = PROJECT_ROOT / "datasets/Hydrangea/fix_repo" / instance_id
    target = root / "datasets/Hydrangea/fix_repo" / instance_id
    target.parent.mkdir(parents=True)
    copy_isolated_tree(source, target)
    (target / "run.log").unlink(missing_ok=True)
    (target / "patch.diff").write_text(patch, encoding="utf-8")


def run_hydrangea(instance_id, repo_path, patch, timeout):
    runner = PROJECT_ROOT / "scripts/reproduce" / f"reproduce_Hydrangea_{instance_id}.py"
    if not runner.is_file():
        raise ValueError(f"Missing Hydrangea reproduction runner: {runner}")
    runner_instance, environment = read_runner_environment(runner)
    if runner_instance != instance_id:
        raise ValueError(f"Hydrangea runner instance mismatch: {runner_instance} != {instance_id}")
    python = environment / ".venv/bin/python"
    nltk_data = environment / "nltk_data"
    source_test = PROJECT_ROOT / "datasets/Hydrangea/fix_repo" / instance_id / "tests/test.py"
    missing = [str(path) for path in (python, source_test) if not path.is_file()]
    if missing:
        raise ValueError(
            f"Hydrangea reproduction environment is incomplete for {instance_id}: {missing}; "
            "run scripts/reproduce/install_Hydrangea_environments.py"
        )

    with tempfile.TemporaryDirectory(prefix=f"lidl-patch-{instance_id}-") as temp_dir:
        isolated_root = Path(temp_dir) / "root"
        isolated_runner = (
            isolated_root
            / "scripts/reproduce"
            / f"reproduce_Hydrangea_{instance_id}.py"
        )
        isolated_runner.parent.mkdir(parents=True)
        runner_text = runner.read_text(encoding="utf-8")
        runner_text, replacements = FIXED_PATCHED_REPO.subn(
            'PATCHED_REPO = ROOT / "datasets/Hydrangea/patched_repo" / INSTANCE_ID',
            runner_text,
        )
        if replacements > 1:
            raise ValueError(f"Hydrangea runner has multiple fixed patched workspaces: {runner}")
        isolated_runner.write_text(runner_text, encoding="utf-8")
        isolated_repo = isolated_root / "datasets/Hydrangea/repo" / instance_id
        isolated_repo.parent.mkdir(parents=True)
        copy_isolated_tree(repo_path, isolated_repo)
        isolated_environment = (
            isolated_root
            / "datasets/Hydrangea/environments"
            / environment.name
        )
        isolated_environment.mkdir(parents=True)
        isolated_environment.joinpath(".venv").symlink_to(
            environment / ".venv", target_is_directory=True
        )
        copy_hydrangea_harness(instance_id, patch, isolated_root)
        defect_source = PROJECT_ROOT / "datasets/Hydrangea/hydrangea_defect.json"
        shutil.copy2(
            defect_source,
            isolated_root / "datasets/Hydrangea/hydrangea_defect.json",
        )

        workspace = isolated_environment / "workspaces" / instance_id
        workspace.parent.mkdir(parents=True)
        copy_isolated_tree(isolated_repo, workspace)
        runner_text = re.sub(r"^(WORKSPACE|PATCHED_REPO) = .+$", lambda match: f'{match[1]} = ENVIRONMENT / "workspaces" / INSTANCE_ID', runner_text, flags=re.MULTILINE)
        isolated_runner.write_text(runner_text, encoding="utf-8")
        apply_result = subprocess.run(
            ["git", "apply", str(
                isolated_root
                / "datasets/Hydrangea/fix_repo"
                / instance_id
                / "patch.diff"
            )],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        if apply_result.returncode != 0:
            return "invalid_patch", {
                "backend": "hydrangea_reproduce_runner",
                "runner": str(runner.relative_to(PROJECT_ROOT)),
                "command": None,
                "cwd": ".",
                "apply": {
                    "returncode": apply_result.returncode,
                    "stdout": apply_result.stdout,
                    "stderr": apply_result.stderr,
                },
                "execution": None,
            }
        arguments = ["--runtime", "local"]
        if "--expect" in runner_text:
            arguments = ["--expect", "pass", *arguments]
        command = [str(python), str(isolated_runner), *arguments]
        test = {
            "backend": "hydrangea_reproduce_runner",
            "runner": str(runner.relative_to(PROJECT_ROOT)),
            "command": command,
            "cwd": ".",
            "execution": None,
        }
        try:
            with hydrangea_instance_lock(instance_id):
                environment = safe_environment()
                # LibreOffice uses XDG_CONFIG_HOME for its per-user profile and IPC lock.
                config_home = isolated_root / "xdg_config"
                config_home.mkdir(exist_ok=True)
                environment["XDG_CONFIG_HOME"] = str(config_home)
                if nltk_data.is_dir():
                    environment["NLTK_DATA"] = str(nltk_data)
                environment["PYTEST_ADDOPTS"] = " ".join(
                    item
                    for item in (
                        environment.get("PYTEST_ADDOPTS", ""),
                        "--rootdir=.",
                    )
                    if item
                )
                process = subprocess.Popen(
                    command,
                    cwd=isolated_root,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    logger.warning(f"Patch test timed out: {error}")
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError as kill_error:
                        logger.warning(f"Hydrangea process group already exited during timeout cleanup: {kill_error}")
                    final_stdout, final_stderr = process.communicate()
                    error.stdout = final_stdout or error.stdout
                    error.stderr = final_stderr or error.stderr
                    raise
            isolated_log = (
                isolated_root
                / "datasets/Hydrangea/fix_repo"
                / instance_id
                / "run.log"
            )
            test["execution"] = {
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
            }
            test["run_log"] = (
                isolated_log.read_text(encoding="utf-8")
                if isolated_log.is_file()
                else None
            )
            return (
                "resolved" if process.returncode == 0 else "unresolved"
            ), test
        except subprocess.TimeoutExpired as error:
            logger.warning(f"Patch test timed out: {error}")
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
            test["execution"] = {
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
            }
            return "unresolved", test


def agentissue_tag(instance_id):
    project, issue = instance_id.split("__", 1)
    issue_number = issue.rsplit("-", 1)[-1]
    return f"{project.lower()}_{issue_number}"


@lru_cache(maxsize=64)
def image_environment(image):
    command = ["docker", "image", "inspect", image, "--format", "{{.Os}}/{{.Architecture}}"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    host = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), platform.machine())
    details = {"command": command, "returncode": result.returncode, "stdout": result.stdout,
               "stderr": result.stderr, "host_architecture": host}
    if result.returncode != 0:
        details["error"] = "image_unavailable"
    elif result.stdout.strip() != f"linux/{host}":
        details["error"] = "image_architecture_mismatch"
    return details


def failure_reason(result, generation_status=None):
    test = result.get("test") or {}
    execution = test.get("execution") or {}
    text = str(execution.get("stdout", "")) + str(execution.get("stderr", "")) + str(test.get("run_log", ""))
    if "LIDL_BASELINE_ALREADY_ACCEPTS_FIELD" in text:
        return "baseline_test_failed"
    if "KeyError: 'OPENAI_API_KEY'" in text:
        return "setup_failed"
    if "Falling back to mock implementation" in text:
        return "mock_fallback_used"
    if "LIDL_HARNESS_IMPORT_FAILED" in text:
        return "test_import_failed"
    if "Resource" in text and any(name in text for name in ("punkt_tab", "averaged_perceptron_tagger_eng")) and "not found" in text:
        return "nltk_resource_missing"
    if "ERROR collecting" in text and "NameError: name 'ContextItem' is not defined" in text:
        return "harness_annotation_error"
    if "patch: command not found" in text:
        return "setup_failed"
    if execution.get("returncode") in {-2, -15, 130, 143}:
        return "execution_interrupted"
    if execution.get("phase") == "setup" and execution.get("timed_out"):
        return "setup_timeout"
    if execution.get("phase") == "setup" and execution.get("returncode") != 0:
        if re.search(r"Hunk #\d+ FAILED|patch failed|malformed patch", text, re.IGNORECASE):
            return "patch_apply_error"
        return "setup_failed"
    if execution.get("phase") in {"create", "start"} and execution.get("returncode") != 0:
        return "setup_failed"
    if execution.get("timed_out") and test.get("backend") == "agentissue_docker" and "Testing PATCHED" not in text and any(marker in text for marker in ("Downloading ", "Installing collected packages", "Building wheels")):
        return "setup_timeout"
    if "ModuleNotFoundError" in text and ("ERROR collecting" in text or "expected 1" in text):
        return "test_dependency_missing"
    if "local original returned" in text or "original tests exited" in text:
        return "baseline_test_failed"
    if "No such file or directory" in text and ("Traceback" in text or "can't open file" in text):
        return "harness_path_error"
    if "exec format error" in text:
        return "image_architecture_mismatch"
    if result.get("status") == "environment_error":
        return result.get("failure_reason") or (test.get("preflight") or {}).get("error", "environment_error")
    if result.get("failure_reason") == "diff_format_error":
        return "diff_format_error"
    validation = result.get("validation") or {}
    if result.get("status") == "invalid_patch":
        if generation_status == "empty_patch":
            return "empty_patch"
        if generation_status == "failed":
            return "generation_error"
        if validation.get("returncode") not in (None, 0):
            return "diff_format_error" if "corrupt patch" in validation.get("stderr", "") else "patch_apply_error"
        if validation.get("syntax_errors"):
            return "syntax_error"
        return "policy_rejection" if validation else "invalid_patch"
    if result.get("status") == "missing_patch":
        return "missing_localization" if generation_status == "missing_localization" else "missing_patch"
    if execution.get("timed_out"):
        return "test_timeout"
    return "test_failed" if result.get("status") == "unresolved" else None


def run_agentissue(instance_id, patch, timeout):
    with DOCKER_SLOTS:
        return run_agentissue_isolated(instance_id, patch, timeout)


def run_agentissue_isolated(instance_id, patch, timeout):
    tag = agentissue_tag(instance_id)
    image = IMAGE_OVERRIDES.get(tag, f"alfin06/agentissue-bench:{tag}")
    preflight = image_environment(image)
    test = {"backend": "agentissue_docker", "image": image, "preflight": preflight, "phases": [], "execution": None}
    if preflight.get("error"):
        logger.error(f"Cannot evaluate {instance_id}: {preflight}")
        return "environment_error", test
    container = f"lidl-eval-{tag}-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix=f"lidl-patch-{instance_id}-") as temp_dir:
        patch_dir = Path(temp_dir) / tag
        patch_dir.mkdir()
        (patch_dir / "patch.patch").write_text(patch, encoding="utf-8")
        create = ["docker", "create", "--name", container, "--entrypoint", "bash", "-v", f"{temp_dir}:/patches:ro", "-e", "OPENAI_API_KEY", "-e", "OPENAI_API_BASE"]
        if tag == "agixt_1369":
            create.extend(["--network", "host"])
        dataset = next(item for item in DATASETS if item["name"] == "AgentIssue-Bench")
        instance = next(item for item in read_json(dataset["dataset_json"]) if item["instance_id"] == instance_id)
        create.extend(["-e", f"BUGGY_COMMIT={instance['base_commit']}", image, "-c", "while true; do sleep 3600; done"])
        phases = [("create", create, 120), ("start", ["docker", "start", container], 120),
                  ("setup", ["docker", "exec", container, "/usr/local/bin/run_test_entrypoint.sh", "apply_patch", f"/patches/{tag}/patch.patch"], 1200),
                  ("test", ["docker", "exec", container, "/usr/local/bin/run_test_entrypoint.sh", "test_patched"], timeout)]
        if tag.startswith("chatdev_") or tag in {"autogen_3361", "autogen_4785", "autogen_5007"}:
            install = "pip install --no-deps -e ."
            if tag.startswith("chatdev_"):
                install = 'if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install --no-deps -e .; else export PYTHONPATH="$source_dir:$PYTHONPATH"; fi'
            elif tag in {"autogen_4785", "autogen_5007"}:
                install = "pip install -e python/packages/autogen-core -e 'python/packages/autogen-ext[openai]'"
            elif tag == "autogen_3361":
                install = "pip install 'numpy<2,>=1.17.0' && pip install -e ."
            setup_fix = (
                "from pathlib import Path; p=Path('/usr/local/bin/run_test_entrypoint.sh'); s=p.read_text(); "
                "assert 'pip install --no-deps -e .' in s, 'Unexpected setup script'; "
                f"s=s.replace('pip install --no-deps -e .', {install!r}); p.write_text(s)"
            )
            phases.insert(2, ("setup", ["docker", "exec", container, "python", "-c", setup_fix], 30))
        if tag.startswith("chatdev_"):
            hook = PROJECT_ROOT / "scripts/paper_chatdev_response_check.py"
            shutil.copy2(hook, patch_dir / "response_check.py")
            test["hook_sha256"] = hashlib.sha256(hook.read_bytes()).hexdigest()
            test["test_scope"] = "offline_response_conversion_with_normal_response_control"
            test["success_criterion"] = "patched_test_pass_only; baseline failure not required"
            field = "refusal" if tag == "chatdev_413" else "audio"
            phases[-1] = ("test", ["docker", "exec", "--workdir", "/app/source_code_buggy", container, "python", f"/patches/{tag}/response_check.py", "--repo", "/app/source_code_buggy", "--field", field], timeout)
        if tag == "agixt_1026":
            phases.insert(-1, ("setup", ["docker", "exec", container, "python", "-m", "pip", "install", "PyJWT==2.8.0", "agixtsdk==0.0.28", "python-dotenv==1.0.0", "fastapi==0.103.2", "PyYAML==6.0.1", "setuptools<81"], 300))
            harness_fix = (
                "from pathlib import Path; "
                "p=Path('/opt/reproduce.py'); s=p.read_text(); "
                "needle='            print(\"Falling back to mock implementation...\")'; "
                "assert s.count(needle)==1, 'Unexpected upstream harness'; "
                "s=s.replace(needle, '            print(\"LIDL_HARNESS_IMPORT_FAILED\"); return 2'); "
                "s=s.replace('sys.path.insert(0, agixt_dir)', "
                "'sys.path.insert(0, agixt_dir)\\n        sys.path.insert(0, os.path.join(agixt_dir, \"agixt\"))'); "
                "p.write_text(s)"
            )
            phases.insert(-1, ("setup", ["docker", "exec", container, "python", "-c", harness_fix], 30))
        try:
            for phase, command, budget in phases:
                started = time.monotonic()
                try:
                    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=budget, check=False)
                    execution = {"phase": phase, "command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "timed_out": False}
                except subprocess.TimeoutExpired as error:
                    logger.warning(f"AgentIssue {phase} timed out for {instance_id}: {error}")
                    execution = {"phase": phase, "command": command, "returncode": None,
                                 "stdout": error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or "",
                                 "stderr": error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or "", "timed_out": True}
                execution["seconds"] = time.monotonic() - started
                test["phases"].append(execution)
                test["execution"] = execution
                if execution["returncode"] != 0:
                    return ("unresolved" if phase == "test" else "environment_error"), test
            passed = tag.startswith("chatdev_") or ("FAILED" not in execution["stdout"] and any(marker in execution["stdout"] for marker in AGENTISSUE_SUCCESS))
            return ("resolved" if passed else "unresolved"), test
        finally:
            try:
                cleanup = subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True, timeout=120, check=False)
                test["cleanup"] = {"returncode": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr}
                if cleanup.returncode:
                    logger.warning(f"Container cleanup failed, preserving evaluation: {container}: {cleanup.stderr}")
            except subprocess.TimeoutExpired as error:
                logger.warning(f"Container cleanup timed out, preserving evaluation: {container}: {error}")
                test["cleanup"] = {"container": container, "timed_out": True,
                                   "stdout": error.stdout.decode(errors='replace') if isinstance(error.stdout, bytes) else error.stdout,
                                   "stderr": error.stderr.decode(errors='replace') if isinstance(error.stderr, bytes) else error.stderr}


def evaluate_unlocked(output_path, overwrite, timeout):
    result_dir = output_path.parent
    result_path = result_dir / "patch_result.json"
    if output_path.name == "trace.json" and not overwrite:
        previous = evaluation_record(result_dir)
        generation_status = read_json(output_path).get("status")
        reason = failure_reason(previous, generation_status)
        patch_path = result_dir / "patch.diff"
        patch_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest() if patch_path.exists() else None
        if previous.get("status") != "not_evaluated" and previous.get("patch_sha256", patch_hash) == patch_hash:
            updated = {**previous, "failure_reason": reason}
            if reason in ENVIRONMENT_REASONS:
                updated["status"] = "environment_error"
            if updated != previous:
                save_evaluation(result_dir, updated)
            return updated
    if output_path.name != "trace.json" and result_path.is_file() and not overwrite:
        patch_path = result_dir / "patch.diff"
        if not patch_path.is_file() or patch_path.stat().st_mtime_ns <= result_path.stat().st_mtime_ns:
            return read_json(result_path)
        logger.warning(f"Patch changed since its last evaluation: {result_dir}")

    repair_output = read_json(output_path)
    localization_path = result_dir / "localization.json"
    if not localization_path.is_file() and output_path.name != "trace.json":
        raise ValueError(f"Missing localization.json beside {output_path}")
    localization = localization_record(result_dir)
    dataset = localization["dataset"]
    instance_id = localization["instance_id"]
    base = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "evaluation_version": EVALUATION_VERSION,
            "patch_sha256": hashlib.sha256((result_dir / "patch.diff").read_bytes()).hexdigest() if (result_dir / "patch.diff").exists() else None}
    generation_status = repair_output.get("status")
    if generation_status != "patch_generated":
        status = "missing_patch" if generation_status == "missing_localization" else "invalid_patch"
        result = {**base, "status": status, "test": None}
    else:
        patch = read_patch(result_dir, repair_output)
        if patch is None:
            result = {**base, "status": "missing_patch", "test": None}
        else:
            repo_path = Path(localization["repo_path"])
            if not repo_path.is_dir():
                raise ValueError(f"Repository directory does not exist: {repo_path}")
            instance = repair_output["instance"] if "instance" in repair_output else read_json(result_dir / "input_raw.json")
            try:
                patch, normalization = prepare_patch(patch)
            except ValueError as error:
                logger.warning(f"Malformed patch for {instance_id}: {error}")
                result = {**base, "status": "invalid_patch", "failure_reason": "diff_format_error", "error": str(error), "test": None}
                save_evaluation(result_dir, result)
                return result
            base["normalization"] = normalization
            validation = validate_patch(instance, repo_path, patch)
            if not validation["valid"]:
                result = {**base, "status": "invalid_patch", "validation": validation, "test": None}
                result["failure_reason"] = failure_reason(result, generation_status)
                save_evaluation(result_dir, result)
                return result
            if dataset == "Hydrangea":
                status, test = run_hydrangea(instance_id, repo_path, patch, timeout)
            elif dataset == "AgentIssue-Bench":
                status, test = run_agentissue(instance_id, patch, timeout)
            else:
                raise ValueError(f"Unsupported repair dataset: {dataset}")
            result = {**base, "status": status, "test": test}
    result["failure_reason"] = failure_reason(result, generation_status)
    if result["failure_reason"] in ENVIRONMENT_REASONS:
        result["status"] = "environment_error"
    elif result["failure_reason"] == "patch_apply_error":
        result["status"] = "invalid_patch"
    save_evaluation(result_dir, result)
    return result



def evaluate_case(result_dir, overwrite, timeout):
    lock_fd = os.open(result_dir, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            started = time.monotonic()
            result = evaluate_unlocked(generation_path(result_dir), overwrite, timeout)
            logger.info(f"Evaluation elapsed {time.monotonic() - started:.2f}s: {result_dir}")
            logger.info(f"Patch evaluation {result['status']}: {result_dir}")
            return result
        except Exception as error:
            logger.error(f"Patch evaluation failed for {result_dir}: {error}")
            result = {"status": "error", "error": str(error), "traceback": traceback.format_exc()}
            save_evaluation(result_dir, result)
            return result
    finally:
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate generated patches, then extract repair metrics")
    parser.add_argument("stage", choices=("evaluate", "extract"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300, help="Test-only time budget; container setup has a separate 1200-second budget")
    parser.add_argument("--image-map", type=Path, help="JSON mapping benchmark image tags to compatible local images")
    parser.add_argument("--retry-harness-fixes", action="store_true", help="Retry only annotation, NLTK-resource and mock-fallback issues; leave image blockers untouched")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-evaluation-fixes", action="store_true", help="Retry affected failures from older evaluator versions, including interrupted historical successes")
    parser.add_argument("--dry-run", action="store_true", help="List selected evaluation paths without executing patches or tests")
    parser.add_argument("--retry-environment-errors", action="store_true", help="Re-evaluate only saved environment failures using the existing patch")
    parser.add_argument("repair_paths", nargs="*", type=Path)
    parser.add_argument("--repair-model", default=DEFAULT_MODEL)
    parser.add_argument("--localization-model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/paper_results")
    parser.add_argument("--limit", type=int, help="Validate with the first N instances of each dataset")
    args = parser.parse_args()
    if sum((args.overwrite, args.retry_environment_errors, args.retry_evaluation_fixes, args.retry_harness_fixes)) > 1:
        parser.error("Choose only one of --overwrite, --retry-environment-errors, --retry-evaluation-fixes")
    if args.repair_model != DEFAULT_MODEL or args.localization_model != DEFAULT_MODEL:
        parser.error("repair and localization models must both be gpt-5.6-terra")

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    try:
        if args.image_map:
            mapping = read_json(args.image_map)
            if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
                raise ValueError("--image-map must contain image-tag to image-name string pairs")
            IMAGE_OVERRIDES.update(mapping)
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.max_workers < 1 or args.timeout < 1:
            parser.error("Workers and timeout must be positive")
        if args.stage == "evaluate":
            allowed = set()
            for dataset in DATASETS:
                instances = read_json(dataset["dataset_json"])
                if args.limit is not None:
                    logger.warning(f"Evaluating first {args.limit} real cases of {dataset['name']}")
                    instances = instances[:args.limit]
                allowed.update((dataset["name"], item["instance_id"]) for item in instances)
            paths = []
            for path in discover_instances(args.repair_paths, args.repair_model, args.localization_model):
                metadata = localization_record(path)
                if (metadata["dataset"], metadata["instance_id"]) in allowed:
                    if args.retry_environment_errors:
                        previous = evaluation_record(path)
                        if previous.get("status") not in {"environment_error", "error"} and failure_reason(previous) not in ENVIRONMENT_REASONS:
                            continue
                    if args.retry_evaluation_fixes:
                        previous = evaluation_record(path)
                        generation = read_json(generation_path(path))
                        reason = failure_reason(previous, generation.get("status"))
                        if (previous.get("status") == "resolved" and reason != "mock_fallback_used") or previous.get("evaluation_version", 0) >= EVALUATION_VERSION:
                            continue
                        if generation.get("status") != "patch_generated":
                            continue
                        if reason not in ENVIRONMENT_REASONS | {"diff_format_error", "patch_apply_error", "test_timeout"} and not any(item.get("status") == "resolved" for item in generation.get("evaluation_history", [])):
                            continue
                    if args.retry_harness_fixes:
                        previous = evaluation_record(path)
                        reason = failure_reason(previous)
                        document_profile_fix = metadata["instance_id"] in {"AgentJ-WR__Private-GPT-Flask-0", "REFRIED-BoolEANS__privatgpt-gui-0"} and previous.get("evaluation_version", 0) < 5 and previous.get("status") == "unresolved"
                        if not document_profile_fix and reason not in {"nltk_resource_missing", "harness_annotation_error", "mock_fallback_used", "test_import_failed"}:
                            continue
                    paths.append(path)
            logger.info(f"Evaluating {len(paths)} patches; first two: {paths[:2]}")
            if args.dry_run:
                for path in paths:
                    logger.info(f"Would evaluate: {path}")
                return 0
            errors = 0
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(evaluate_case, path, args.overwrite or args.retry_environment_errors or args.retry_evaluation_fixes or args.retry_harness_fixes, args.timeout) for path in paths]
                for future in as_completed(futures):
                    errors += future.result()["status"] == "error"
            return int(errors > 0)
        rows = collect_benchmark_rows(args.repair_paths, args.repair_model, args.localization_model, args.limit)
        logger.info(f"Selected {len(rows)} benchmark-method instances; first two: {rows[:2]}")
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["dataset"], row["repair_model"], row["approach"], row["localization_model"])].append(row)
        summaries = [summarize(rows, "overall", "all")]
        summaries.extend(
            summarize(group_rows, "method", "/".join(str(value) for value in key))
            for key, group_rows in sorted(grouped.items())
        )
        summaries.extend(summarize([row for row in rows if row["approach"] == method], "approach", method) for method in METHODS)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "repair_instances.csv", rows)
        write_csv(output_dir / "repair_summary.csv", summaries)
        write_repair_table(output_dir, summaries)
        logger.info(f"Aggregated {len(rows)} repair instances into {output_dir}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        logger.error(f"Repair result aggregation failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import fcntl
import json
import os
import re
import shutil
from dotenv import load_dotenv
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger
from litellm import token_counter

from paper_results import (
    DATASETS,
    collect_functions,
    limit_ranked_lines,
    normalize_path,
    parse_agentless_lines,
    parse_autocoderover_lines,
    parse_sweagent_lines,
    read_autocoderover_locations,
    read_json,
    read_jsonl_record,
)
from utils import append_jsonl, atomic_json, call_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CodeSpan:
    file_path: str
    start_line: int
    end_line: int
    text: str


@lru_cache(maxsize=128)
def read_file_lines(root_dir, file_path):
    full_path = Path(root_dir) / file_path
    with open(full_path, encoding="utf-8", newline="") as source_file:
        return source_file.read().splitlines(keepends=True)


def get_code_span(root_dir, file_path, start_line, end_line):
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")
    lines = read_file_lines(str(Path(root_dir)), file_path)
    if start_line > len(lines):
        raise ValueError(f"start_line {start_line} beyond end of file with {len(lines)} lines")
    return CodeSpan(
        file_path=file_path,
        start_line=start_line,
        end_line=min(end_line, len(lines)),
        text="".join(lines[start_line - 1 : end_line]),
    )


RESULT_ROOTS = {
    "results_agentless": "Agentless",
    "results_agentless_repograph": "Agentless*",
    "results_autocoderover": "AutoCodeRover",
    "results_repograph": "RepoGraph",
    "results_locagent": "LocAgent",
    "results_sweagent": "SWE-agent",
    "results_sweagent_repograph": "SWE-agent*",
    "results_codex": "Codex",
    "results_main": "LIDL",
}
APPROACH_ROOTS = {approach: root for root, approach in RESULT_ROOTS.items()}
PATCH_BLOCK = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
MODEL_SETTING_LINE = re.compile(r"^[+-].*\bmodel\s*=", re.MULTILINE | re.IGNORECASE)
LOGGING_LINE = re.compile(r"^[+-].*\b(?:logger|logging)\.", re.IGNORECASE)
OUTPUT_REPLACEMENT_LINE = re.compile(r"^[+].*\.replace\s*\(", re.IGNORECASE)
BEHAVIORAL_ISSUE = re.compile(
    r"answer|admit|assistant|character|contradict|identity|language model|response",
    re.IGNORECASE,
)


def detect_result(result_path):
    parts = result_path.parts
    for root_name, approach in RESULT_ROOTS.items():
        if root_name not in parts:
            continue
        root_index = parts.index(root_name)
        if len(parts) <= root_index + 2:
            raise ValueError(f"Result path does not contain dataset and model: {result_path}")
        dataset = parts[root_index + 1]
        model = parts[root_index + 2]
        if approach.startswith("SWE-agent") and model.startswith("openai-"):
            model = model[len("openai-") :]
        return approach, dataset, model
    raise ValueError(f"Unsupported result path: {result_path}")


def infer_autocoderover_instance(result_path):
    meta_path = result_path / "meta.json"
    if meta_path.exists():
        instance_id = (read_json(meta_path).get("task_info") or {}).get("instance_id")
        if instance_id:
            return instance_id
    for marker in ("_20", "_19"):
        if marker in result_path.name:
            return result_path.name.split(marker, 1)[0]
    return result_path.name


def latest_trajectory(result_path):
    trajectories = list(result_path.rglob("*.traj"))
    return max(trajectories, key=lambda path: path.stat().st_mtime) if trajectories else None


def infer_instance_id(result_path, approach, requested_instance_id):
    if requested_instance_id:
        return requested_instance_id
    if approach.startswith("Agentless") or approach in {"Codex", "LIDL", "RepoGraph"}:
        return result_path.name
    if approach == "AutoCodeRover":
        return infer_autocoderover_instance(result_path)
    if approach == "LocAgent":
        raise ValueError("--instance-id is required for a LocAgent model result directory")
    trajectory_path = latest_trajectory(result_path)
    if trajectory_path is None:
        raise ValueError(f"No SWE-agent trajectory found under {result_path}")
    return result_path.parent.name if trajectory_path.parent == result_path else trajectory_path.parent.parent.name


def read_agentless_locations(result_path, instance_id):
    source_path = result_path / "edit_location_individual/loc_merged_0-0_outputs.jsonl"
    record = read_jsonl_record(source_path, instance_id)
    grouped = parse_agentless_lines((record or {}).get("found_edit_locs"), instance_id)
    ranked_lines = [(path, line_number) for path, line_numbers in grouped.items() for line_number in line_numbers]
    return ranked_lines, source_path if record else None


def read_autocoderover_result(result_path, instance_id):
    for output_dir in sorted(
        (path for path in result_path.glob("output_*") if path.is_dir()),
        reverse=True,
    ):
        locations = read_autocoderover_locations(output_dir)
        if not locations:
            continue
        source_path = next(
            (
                output_dir / "search" / name
                for name in ("bug_locations_after_process.json", "bug_locations_before_process.json")
                if (output_dir / "search" / name).exists()
            ),
            None,
        )
        return parse_autocoderover_lines(locations, instance_id), source_path
    return [], None


def read_locagent_result(result_path, instance_id):
    source_path = result_path / "merged_loc_outputs_mrr.jsonl"
    record = read_jsonl_record(source_path, instance_id)
    ranked_lines = []
    for location in (record or {}).get("line_locations", []):
        if not isinstance(location, (list, tuple)) or len(location) < 3:
            continue
        path = normalize_path(location[0], instance_id)
        line_number = location[2]
        if path and isinstance(line_number, int) and not isinstance(line_number, bool):
            ranked_lines.append((path, line_number))
    return ranked_lines, source_path if record else None


def read_sweagent_result(result_path, instance_id):
    source_path = latest_trajectory(result_path)
    if source_path is None:
        return [], None
    return parse_sweagent_lines(read_json(source_path), instance_id), source_path


def read_output_locations(result_path, instance_id):
    source_path = result_path / "output.json"
    if not source_path.is_file():
        return [], None
    output = read_json(source_path)
    ranked_lines = [
        (normalize_path(item.get("path", ""), instance_id), item.get("line"))
        for item in output.get("ranked_locations", [])
        if isinstance(item, dict)
        and item.get("path")
        and isinstance(item.get("line"), int)
        and not isinstance(item.get("line"), bool)
    ]
    return ranked_lines, source_path


def read_ranked_lines(result_path, approach, instance_id):
    if approach.startswith("Agentless"):
        return read_agentless_locations(result_path, instance_id)
    if approach == "AutoCodeRover":
        return read_autocoderover_result(result_path, instance_id)
    if approach == "LocAgent":
        return read_locagent_result(result_path, instance_id)
    if approach in {"Codex", "LIDL", "RepoGraph"}:
        return read_output_locations(result_path, instance_id)
    return read_sweagent_result(result_path, instance_id)


def infer_repo_path(dataset, instance_id, requested_repo_path):
    if requested_repo_path:
        return requested_repo_path.expanduser().resolve()
    return PROJECT_ROOT / "datasets" / dataset / "repo" / instance_id


def enclosing_function(repo_path, relative_path, line_number):
    functions = collect_functions(repo_path / relative_path)
    candidates = [item for item in functions if item[1] <= line_number <= item[2]]
    selected = min(candidates, key=lambda item: item[2] - item[1], default=None)
    return selected[0] if selected else None


def collect_localization(result_path, instance_id=None, repo_path=None, limit=10):
    result_path = result_path.expanduser().resolve()
    if not result_path.is_dir():
        raise ValueError(f"Result directory does not exist: {result_path}")
    approach, dataset, model = detect_result(result_path)
    instance_id = infer_instance_id(result_path, approach, instance_id)
    repo_path = infer_repo_path(dataset, instance_id, repo_path)
    if not repo_path.is_dir():
        raise ValueError(f"Repository directory does not exist: {repo_path}")

    ranked_lines, source_path = read_ranked_lines(result_path, approach, instance_id)
    ranked_lines = limit_ranked_lines(ranked_lines, limit)
    locations = []
    invalid_paths = []
    for rank, (relative_path, line_number) in enumerate(ranked_lines, 1):
        source_file = repo_path / relative_path
        if not source_file.is_file():
            invalid_paths.append(relative_path)
        locations.append(
            {
                "rank": rank,
                "file": relative_path,
                "function": enclosing_function(repo_path, relative_path, line_number),
                "line": line_number,
            }
        )
    return {
        "dataset": dataset,
        "model": model,
        "approach": approach,
        "instance_id": instance_id,
        "status": "completed" if locations else "missing_or_incomplete",
        "result_path": str(result_path),
        "prediction_source": str(source_path) if source_path else None,
        "repo_path": str(repo_path),
        "locations": locations,
        "invalid_prediction_paths": list(dict.fromkeys(invalid_paths)),
    }


def read_instance(dataset, instance_id):
    dataset_config = next((item for item in DATASETS if item["name"] == dataset), None)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    instances = json.loads(dataset_config["dataset_json"].read_text(encoding="utf-8"))
    instance = next((item for item in instances if item["instance_id"] == instance_id), None)
    if instance is None:
        raise ValueError(f"Instance not found in {dataset}: {instance_id}")
    return instance


def enclosing_span(repo_path, location, context_lines):
    relative_path = location["file"]
    line_number = location["line"]
    source_path = repo_path / relative_path
    if not source_path.is_file():
        return None
    functions = collect_functions(source_path)
    candidates = [item for item in functions if item[1] <= line_number <= item[2]]
    selected = min(candidates, key=lambda item: item[2] - item[1], default=None)
    line_count = len(source_path.read_text(encoding="utf-8", errors="replace").splitlines())
    if selected:
        start_line = max(1, selected[1] - context_lines)
        end_line = min(line_count, selected[2] + context_lines)
    else:
        start_line = max(1, line_number - context_lines)
        end_line = min(line_count, line_number + context_lines)
    span = get_code_span(repo_path, relative_path, start_line, end_line)
    return {
        "file": relative_path,
        "start_line": span.start_line,
        "end_line": span.end_line,
        "text": span.text,
    }


def build_context(repo_path, localization, omitted):
    spans = []
    paths = list(dict.fromkeys(item["file"] for item in localization["locations"]))
    for path in paths:
        source = repo_path / path
        if Path(path).is_absolute() or not source.resolve().is_relative_to(repo_path.resolve()) or not source.is_file():
            omitted.append({"file": path, "reason": "invalid_file"})
            logger.warning(f"Skipping invalid localization file: {path}")
            continue
        if len(spans) == 5:
            omitted.append({"file": path, "reason": "top5_file_budget"})
            logger.warning(f"File outside Top-5 context budget: {path}")
            continue
        text = source.read_text(encoding="utf-8")
        spans.append({"file": path, "start_line": 1, "end_line": len(text.splitlines()), "text": text})
    localization["context_files"] = [span["file"] for span in spans]
    return spans


def number_span(span):
    numbered = "\n".join(
        f"{line_number}|{line}" for line_number, line in enumerate(span["text"].splitlines(), span["start_line"])
    )
    return f"### {span['file']}:{span['start_line']}-{span['end_line']}\n" f"```\n{numbered}\n```"


def build_prompt(instance, localization, spans):
    ranked_locations = "\n".join(
        f"{rank}. {span['file']}" for rank, span in enumerate(spans, 1)
    )
    code_context = "\n\n".join(number_span(span) for span in spans)
    example_output = json.dumps(
        {
            "analysis": "The function passes None directly to a list comprehension, which raises TypeError because None is not iterable. Returning an empty list before iteration handles None and preserves the existing name-cleaning behavior for lists.",
            "patch": "diff --git a/app/names.py b/app/names.py\n--- a/app/names.py\n+++ b/app/names.py\n@@ -1,2 +1,4 @@\n def normalize_names(names):\n+    if names is None:\n+        return []\n     return [name.strip() for name in names]\n",
        },
        indent=2,
    )
    return f"""You are an automated program repair assistant. Use the defect description, ranked locations, and code context to produce a minimal, correct production-code patch.

## Requirements
1. Identify the root cause. Treat ranked locations as clues, not confirmed defect locations. Trace the reported failure through the supplied code and fix its cause rather than hide its symptoms.
2. Ground the patch in the supplied code. Use repository-relative paths and existing files, symbols, and APIs. Do not invent thresholds, policies, dependencies, files, symbols, and APIs. Before changing a literal value, confirm its presence and role in the faulty execution.
3. Keep changes minimal and complete. Fix all affected equivalent paths shown in the context. Preserve unrelated behavior, APIs, and code style. Leave tests, generated files, dependency lock files, and unrelated code unchanged. For response-content and identity defects, changes limited to model names, model versions, and dependency versions do not qualify as a fix.
4. Return an applicable unified diff beginning with diff --git and compatible with git apply. Copy unchanged and removed lines exactly from the code context after removing the LINE| prefix. If the supplied evidence cannot support a fix, explain what is missing and return an empty patch string.
5. Return a valid JSON object containing analysis and patch. In analysis, briefly explain the faulty data flow and why the change fixes it. In patch, provide the complete diff. Do not include Markdown fences and text outside the JSON object.

## Example
Input:
<issue>
normalize_names raises TypeError when names is None.
The function should return an empty list for None and continue to strip leading and trailing whitespace from each name in a list.
</issue>

<ranked_localization>
1. app/names.py:2 (normalize_names)
</ranked_localization>

<repository_context>
### app/names.py:1-2
1|def normalize_names(names):
2|    return [name.strip() for name in names]
</repository_context>

Output:
{example_output}

## Actual task
<issue>
{instance['problem_statement']}
</issue>

<ranked_localization>
{ranked_locations}
</ranked_localization>

<repository_context>
{code_context}
</repository_context>"""


def normalize_patch(value):
    patch = str(value or "").strip()
    match = PATCH_BLOCK.fullmatch(patch)
    if match:
        patch = match.group(1).strip()
    if not patch:
        raise ValueError("Model returned an empty patch")
    if not patch.startswith("diff --git "):
        raise ValueError("Model patch is not a git unified diff")
    return f"{patch}\n"


def write_log(output_dir, level, message):
    timestamp = datetime.now(timezone.utc).isoformat()
    with (output_dir / "run.log").open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} [{level}] {message}\n")


def output_dir_for(localization, args, batch_mode):
    output_dir = (
        PROJECT_ROOT
        / "results/results_repair"
        / localization["dataset"]
        / args.model.replace("/", "-")
        / localization["approach"]
        / localization["model"]
        / localization["instance_id"]
    )
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
        if batch_mode:
            output_dir = (
                output_dir
                / localization["dataset"]
                / args.model.replace("/", "-")
                / localization["approach"]
                / localization["model"]
                / localization["instance_id"]
            )
    return output_dir








def process_job_unlocked(result_path, instance_id, args, batch_mode):
    approach, dataset, model = detect_result(result_path)
    identity = {"dataset": dataset, "model": model, "approach": approach,
                "instance_id": infer_instance_id(result_path, approach, instance_id)}
    output_dir = output_dir_for(identity, args, batch_mode)
    output_path = output_dir / "trace.json"
    patch_path = output_dir / "patch.diff"
    if output_path.is_file() and not args.overwrite:
        previous = read_json(output_path)
        status = previous.get("status")
        if previous.get("context_policy") != "top5_complete_files":
            raise ValueError(f"Existing repair uses another context policy; archive it before running: {output_dir}")
        if status not in {"running", "prepared"} and (args.prompt_only or status != "prompt_only"):
            logger.info(f"Skipping existing repair: {identity}")
            return identity["instance_id"], status in {"prompt_only", "patch_generated"}
    record = {"schema_version": 2, "context_policy": "top5_complete_files", "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "localization": identity, "input": None, "api_records": [], "context_omissions": []}
    atomic_json(output_path, record)
    patch_path.write_text("", encoding="utf-8")

    def save_api_record(api_record):
        record["api_records"].append(api_record)
        atomic_json(output_path, record)

    try:
        localization = collect_localization(result_path, instance_id, args.repo_path, args.location_limit)
        record["localization"] = localization
        instance = read_instance(dataset, identity["instance_id"])
        record["instance"] = instance
        if not localization["locations"]:
            raise ValueError("Localization result contains no valid locations")
        spans = build_context(Path(localization["repo_path"]), localization, record["context_omissions"])
        messages = [{"role": "user", "content": build_prompt(instance, localization, spans)}]
        input_tokens = token_counter(model=args.model, messages=messages)
        record["input"] = {"model": args.model, "api_base": args.api_base, "messages": messages,
                           "max_tokens": args.max_tokens, "temperature": 0,
                           "max_retries": 1, "estimated_input_tokens": input_tokens,
                           "max_input_tokens": args.max_input_tokens}
        record["status"] = "prepared"
        atomic_json(output_path, record)
        if not spans:
            raise ValueError("No localized source context could be loaded")
        if input_tokens > args.max_input_tokens:
            raise ValueError(f"Prompt requires {input_tokens} tokens, exceeding {args.max_input_tokens}")
        if args.prompt_only:
            record["status"] = "prompt_only"
        else:
            response = call_llm(messages, args.model, args.api_base, args.api_key,
                                log_file=save_api_record, log_input=False, temperature=0,
                                max_retries=1, max_tokens=args.max_tokens)
            patch = response.get("patch")
            if not isinstance(patch, str):
                raise ValueError("Response patch must be a string")
            patch_path.write_text(patch, encoding="utf-8")
            record["status"] = "patch_generated" if patch.strip() else "empty_patch"
    except Exception as error:
        logger.error(f"Repair generation failed for {identity}: {error}")
        record["status"] = "missing_localization" if not record["localization"].get("locations") else "failed"
        record["error"] = {"message": str(error), "type": type(error).__name__, "traceback": traceback.format_exc()}
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output_path, record)
    logger.info(f"Repair result {record['status']}: {output_dir}")
    return identity["instance_id"], record["status"] in {"patch_generated", "prompt_only"}


def process_job(result_path, instance_id, args, batch_mode):
    approach, dataset, model = detect_result(result_path)
    localization = {"dataset": dataset, "model": model, "approach": approach,
                    "instance_id": infer_instance_id(result_path, approach, instance_id)}
    output_dir = output_dir_for(localization, args, batch_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(output_dir, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return process_job_unlocked(result_path, instance_id, args, batch_mode)
    finally:
        os.close(lock_fd)


def build_parser():
    parser = argparse.ArgumentParser(description="Generate repair patches from localization results")
    parser.add_argument("result_paths", nargs="*", type=Path, help="Instance or model result directories; default: all standard results")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--repo-path", type=Path, help="Repository path for one single-instance run")
    parser.add_argument("--model", default="gpt-5.6-terra", help="Repair model")
    parser.add_argument(
        "--localization-model",
        default="gpt-5.6-terra",
        help="Localization model; this experiment requires gpt-5.6-terra",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8322/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "sk-1234"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Maximum instances per result directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(location_limit=10, max_tokens=32000, max_input_tokens=700000)
    parser.add_argument("--prompt-only", action="store_true")
    return parser

def main():
    load_dotenv(PROJECT_ROOT / "scripts/.env")
    parser = build_parser()
    args = parser.parse_args()
    if len(args.result_paths) != 1 or (args.instance_ids and len(args.instance_ids) != 1):
        parser.error("Provide one localization result path and at most one --instance-id")
    if args.model != "gpt-5.6-terra" or args.localization_model != args.model:
        parser.error("Repair and localization models must be gpt-5.6-terra")
    args.api_key = os.getenv("LLM_API_KEY") or args.api_key
    path = args.result_paths[0].expanduser().resolve()
    try:
        iid, generated = process_job(path, args.instance_ids[0] if args.instance_ids else None, args, False)
        logger.info(f"Single repair complete: {iid}, generated={generated}")
        return 0
    except Exception as error:
        logger.error(f"Single repair failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

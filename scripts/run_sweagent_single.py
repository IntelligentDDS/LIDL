import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "SWE-agent"
SWEAGENT_BIN = METHOD_DIR / ".venv/bin/sweagent"
DEFAULT_CONFIG = METHOD_DIR / "config/sweagent_localization.yaml"
MAX_LOCATIONS = 10
LOCATION_BLOCK = re.compile(r"LOCATIONS\s*\n(.*?)\nEND_LOCATIONS", re.DOTALL | re.IGNORECASE)
LOCATION_LINE = re.compile(
    r"file\s*:\s*(.+?)\s*,\s*function\s*:\s*(.+?)\s*,\s*line\s*:\s*(\d+)",
    re.IGNORECASE,
)


def normalize_model_name(model, api_base):
    providers = {
        "anthropic",
        "azure",
        "bedrock",
        "cohere",
        "gemini",
        "groq",
        "huggingface",
        "openai",
        "together_ai",
        "vertex_ai",
    }
    if "/" in model and model.split("/", 1)[0] in providers:
        return model
    if "api.anthropic.com" in api_base:
        return f"anthropic/{model}"
    return f"openai/{model}"


def parse_locations(text):
    match = LOCATION_BLOCK.search(text)
    if not match:
        return []
    locations = []
    seen = set()
    for line in match.group(1).splitlines():
        location_match = LOCATION_LINE.search(line)
        if not location_match:
            continue
        location = (
            location_match.group(1).strip().removeprefix("./"),
            location_match.group(2).strip(),
            int(location_match.group(3)),
        )
        if location in seen:
            continue
        seen.add(location)
        locations.append(location)
        if len(locations) == MAX_LOCATIONS:
            break
    return locations


def parse_trajectory(trace):
    texts = []
    for item in reversed(trace.get("history", [])):
        content = item.get("content")
        if isinstance(content, str) and "LOCATIONS" in content.upper():
            texts.append(content)
    for step in reversed(trace.get("trajectory", [])):
        for field in ("observation", "response", "action"):
            content = step.get(field)
            if isinstance(content, str) and "LOCATIONS" in content.upper():
                texts.append(content)
    for text in texts:
        locations = parse_locations(text)
        if locations:
            return locations
    return []


def valid_run(run_dir):
    traces = sorted(run_dir.glob("*.traj"))
    if len(traces) != 1:
        return False
    try:
        trace = json.loads(traces[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(f"Invalid trajectory ignored: {traces[0]}: {error}")
        return False
    events_path = traces[0].with_suffix(".llm_events.jsonl")
    return bool(parse_trajectory(trace)) and events_path.is_file() and events_path.stat().st_size > 0


def find_completed_run(instance_dir):
    if not instance_dir.is_dir():
        return None
    for run_dir in sorted(path for path in instance_dir.iterdir() if path.is_dir()):
        if valid_run(run_dir):
            return run_dir
    return None


def call_limit_reached(instance_dir, max_llm_calls):
    marker_path = instance_dir / "llm_call_limit_reached"
    if marker_path.is_file():
        return True
    for trajectory_path in instance_dir.glob("*/*.traj"):
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Invalid trajectory ignored: {trajectory_path}: {error}")
            continue
        info = trajectory.get("info", {})
        exit_status = info.get("exit_status", "")
        api_calls = info.get("model_stats", {}).get("api_calls")
        if "exit_call_limit" in exit_status or (
            "exit_cost" in exit_status and isinstance(api_calls, int) and api_calls >= min(max_llm_calls, 50)
        ):
            return True
    return False


def prepare_config(args, model_name):
    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    model_config = config["agent"]["model"]
    model_config["name"] = model_name
    model_config["api_base"] = args.api_base
    model_config["api_key"] = args.api_key
    model_config["temperature"] = 0.0
    model_config["per_instance_call_limit"] = args.max_llm_calls
    return config


def prepare_repository(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=destination,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["git", "add", "-f", "-A"], cwd=destination, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SWE-agent",
            "-c",
            "user.email=sweagent@local",
            "commit",
            "--quiet",
            "-m",
            "Initial snapshot",
        ],
        cwd=destination,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_instance(instance, args):
    instance_id = instance["instance_id"]
    instance_dir = args.output_base / instance_id
    try:
        completed_run = find_completed_run(instance_dir)
        if call_limit_reached(instance_dir, args.max_llm_calls):
            status = "skipped" if completed_run else "call_limit"
            logger.warning(f"SKIP {instance_id}: terminal {args.max_llm_calls}-call limit already reached")
            return {"instance_id": instance_id, "status": status}
        if completed_run and not args.overwrite:
            logger.info(f"SKIP {instance_id}: {completed_run}")
            return {"instance_id": instance_id, "status": "skipped"}
        if instance_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Incomplete output exists at {instance_dir}; inspect it or use --overwrite")
            shutil.rmtree(instance_dir)

        source_repo = args.repo_base / instance_id
        if not source_repo.is_dir():
            raise FileNotFoundError(f"Repository not found: {source_repo}")
        problem_statement = instance.get("problem_statement")
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError(f"Missing problem_statement for {instance_id}")

        normalized_problem = problem_statement.replace("\r\n", "\n").replace("\r", "\n")
        run_id = hashlib.sha256(normalized_problem.encode()).hexdigest()[:6]
        run_dir = instance_dir / run_id
        run_dir.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix=f"sweagent-{instance_id}-") as temporary_dir:
            temporary_dir = Path(temporary_dir)
            issue_path = temporary_dir / "problem_statement.md"
            config_path = temporary_dir / "config.yaml"
            repository = temporary_dir / instance_id
            issue_path.write_text(problem_statement, encoding="utf-8")
            model_name = normalize_model_name(args.model, args.api_base)
            config_path.write_text(yaml.safe_dump(prepare_config(args, model_name), sort_keys=False), encoding="utf-8")
            prepare_repository(source_repo, repository)

            command = [
                str(SWEAGENT_BIN),
                "run",
                "--config",
                str(config_path),
                "--env.repo.path",
                str(repository),
                "--problem_statement.path",
                str(issue_path),
                "--output_dir",
                str(instance_dir),
                "--env.deployment.image",
                args.docker_image,
            ]
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["SWEAGENT_LLM_EVENTS_PATH"] = str(run_dir / f"{run_id}.llm_events.jsonl")
            environment["SWEAGENT_CALL_LIMIT_MARKER_PATH"] = str(instance_dir / "llm_call_limit_reached")
            logger.info(f"RUN {instance_id}: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                cwd=METHOD_DIR,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("SWE-agent stdout pipe was not created")
            for line in process.stdout:
                print(f"[{instance_id}] {line}", end="", flush=True)
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"SWE-agent exited with code {returncode}")
        if not valid_run(run_dir):
            raise RuntimeError("SWE-agent completed without a valid LOCATIONS block")

        trace = json.loads((run_dir / f"{run_id}.traj").read_text(encoding="utf-8"))
        logger.info(f"DONE {instance_id}: {parse_trajectory(trace)[:2]}")
        return {"instance_id": instance_id, "status": "completed"}
    except Exception as error:
        logger.error(f"FAIL {instance_id}: {type(error).__name__}: {error}\n{traceback.format_exc()}")
        return {"instance_id": instance_id, "status": "failed"}


def select_instances(dataset, instance_ids, limit):
    requested = set(instance_ids or [])
    selected = [item for item in dataset if not requested or item.get("instance_id") in requested]
    if requested:
        missing = requested - {item["instance_id"] for item in selected}
        if missing:
            raise ValueError(f"Instance IDs not found: {sorted(missing)}")
    if limit is not None:
        selected = selected[:limit]
    ids = [item["instance_id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate instance_id values")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Running SWE-agent single data set positioning in parallel")
    parser.add_argument("--dataset-json", "--dataset_json", dest="dataset_json", type=Path, default=PROJECT_ROOT / "datasets/AgentIssue-Bench/agent_issue.json")
    parser.add_argument("--repo-base", "--repo_base", dest="repo_base", type=Path, default=PROJECT_ROOT / "datasets/AgentIssue-Bench/repo")
    parser.add_argument("--output-base", "--output_base", dest="output_base", type=Path)
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", "--api_base", dest="api_base", required=True)
    parser.add_argument("--api-key", "--api_key", dest="api_key", required=True)
    parser.add_argument("--base-config", "--base_config", dest="base_config", type=Path)
    parser.add_argument("--docker-image", default="python:3.11")
    parser.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int, default=1)
    parser.add_argument("--max-llm-calls", "--max_llm_calls", dest="max_llm_calls", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    if not args.api_base.startswith(("http://", "https://")):
        parser.error("--api-base must start with http:// or https://")
    args.dataset_json = args.dataset_json.expanduser().resolve()
    args.repo_base = args.repo_base.expanduser().resolve()
    args.base_config = (args.base_config or DEFAULT_CONFIG).expanduser().resolve()
    if args.output_base is not None:
        args.output_base = args.output_base.expanduser().resolve()
    if not args.dataset_json.is_file():
        parser.error(f"Dataset not found: {args.dataset_json}")
    if not args.repo_base.is_dir():
        parser.error(f"Repository base not found: {args.repo_base}")
    if not args.base_config.is_file():
        parser.error(f"Config not found: {args.base_config}")
    if not SWEAGENT_BIN.is_file():
        parser.error(f"SWE-agent executable not found: {SWEAGENT_BIN}")
    if args.output_base is None:
        dataset_name = args.dataset_json.parent.name
        model_dir = normalize_model_name(args.model, args.api_base).replace("/", "-")
        args.output_base = PROJECT_ROOT / "results/results_sweagent" / dataset_name / model_dir
    return args


def main():
    args = parse_args()
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    instances = select_instances(dataset, args.instance_ids, args.limit)
    if not instances:
        raise ValueError("No instances selected")
    logger.info(f"Selected {len(instances)} instances; first two: {[item['instance_id'] for item in instances[:2]]}")

    results = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(instances))) as executor:
        futures = {executor.submit(run_instance, instance, args): instance["instance_id"] for instance in instances}
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                logger.error(f"Worker crashed for {instance_id}: {error}")
                results.append({"instance_id": instance_id, "status": "failed"})
    counts = {status: sum(item["status"] == status for item in results) for status in sorted({item["status"] for item in results})}
    logger.info(f"Summary: {counts}; first two results: {results[:2]}")
    return 1 if any(item["status"] in {"failed", "call_limit"} for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

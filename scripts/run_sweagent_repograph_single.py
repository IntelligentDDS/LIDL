import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "SWE-agent-RepoGraph"
SWEAGENT_BIN = METHOD_DIR / ".venv/bin/sweagent"
DEFAULT_CONFIG = METHOD_DIR / "config/sweagent_localization_repograph.yaml"
MAX_LOCATIONS = 10
LOCATION_BLOCK = re.compile(r"LOCATIONS\s*\n(.*?)\nEND_LOCATIONS", re.DOTALL | re.IGNORECASE)
LOCATION_LINE = re.compile(
    r"file\s*:\s*(.+?)\s*,\s*function\s*:\s*(.+?)\s*,\s*line\s*:\s*(\d+)",
    re.IGNORECASE,
)


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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_model_name(model, api_base):
    providers = {"anthropic", "azure", "bedrock", "cohere", "gemini", "groq", "huggingface", "openai", "together_ai", "vertex_ai"}
    if "/" in model and model.split("/", 1)[0] in providers:
        return model
    if "api.anthropic.com" in api_base:
        return f"anthropic/{model}"
    return f"openai/{model}"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def find_completed_run(instance_dir):
    for pred_path in sorted(instance_dir.glob("*/*.pred")):
        traj_path = pred_path.with_suffix(".traj")
        if not traj_path.is_file():
            continue
        try:
            prediction = json.loads(pred_path.read_text(encoding="utf-8"))
            trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"WARNING: invalid result ignored: {pred_path}: {error}", file=sys.stderr)
            continue
        events_path = pred_path.with_suffix(".llm_events.jsonl")
        if (
            prediction.get("instance_id")
            and parse_trajectory(trajectory)
            and events_path.is_file()
            and events_path.stat().st_size > 0
        ):
            return pred_path.parent
    return None


def call_limit_reached(instance_dir, max_llm_calls):
    marker_path = instance_dir / "llm_call_limit_reached"
    if marker_path.is_file():
        return True
    for trajectory_path in instance_dir.glob("*/*.traj"):
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"WARNING: invalid result ignored: {trajectory_path}: {error}", file=sys.stderr)
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
    model_config["top_p"] = None
    model_config["per_instance_call_limit"] = args.max_llm_calls
    return config


def run_instance(instance, args):
    instance_id = instance["instance_id"]
    instance_dir = args.output_base / instance_id
    status_path = instance_dir / "status.json"

    try:
        completed_run = find_completed_run(instance_dir)
        if call_limit_reached(instance_dir, args.max_llm_calls):
            result = {
                "instance_id": instance_id,
                "status": "skipped" if completed_run else "call_limit",
                "updated_at": utc_now(),
            }
            write_json(status_path, result)
            print(f"SKIP {instance_id}: terminal {args.max_llm_calls}-call limit already reached")
            return result
        if completed_run and not args.overwrite:
            result = {
                "instance_id": instance_id,
                "status": "skipped",
                "run_dir": str(completed_run),
                "updated_at": utc_now(),
            }
            write_json(status_path, result)
            print(f"SKIP {instance_id}: completed result at {completed_run}")
            return result
        if instance_dir.exists():
            if not args.overwrite:
                message = f"Incomplete output exists at {instance_dir}; inspect it or rerun with --overwrite"
                write_json(status_path, {"instance_id": instance_id, "status": "blocked", "error": message, "updated_at": utc_now()})
                print(f"ERROR {instance_id}: {message}", file=sys.stderr)
                return {"instance_id": instance_id, "status": "blocked", "error": message}
            shutil.rmtree(instance_dir)

        repo_path = args.repo_base / instance_id
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        problem_statement = instance.get("problem_statement")
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            raise ValueError(f"Missing problem_statement for {instance_id}")

        model_name = normalize_model_name(args.model, args.api_base)
        normalized_problem = problem_statement.replace("\r\n", "\n").replace("\r", "\n")
        run_id = hashlib.sha256(normalized_problem.encode()).hexdigest()[:6]
        run_dir = instance_dir / run_id
        input_data = {
            "dataset_json": str(args.dataset_json),
            "instance": instance,
            "repo_path": str(repo_path),
            "model": model_name,
            "api_base": args.api_base,
            "base_config": str(args.base_config),
        }
        write_json(instance_dir / "input_raw.json", input_data)
        write_json(
            status_path,
            {"instance_id": instance_id, "status": "running", "run_dir": str(run_dir), "started_at": utc_now()},
        )

        with tempfile.TemporaryDirectory(prefix=f"sweagent-{instance_id}-") as temporary_dir:
            temporary_dir = Path(temporary_dir)
            issue_path = temporary_dir / "problem_statement.md"
            config_path = temporary_dir / "config.yaml"
            temporary_repo = temporary_dir / "repo"
            shutil.copytree(repo_path, temporary_repo, ignore=shutil.ignore_patterns(".git"))
            subprocess.run(["git", "init", "--quiet"], cwd=temporary_repo, check=True)
            subprocess.run(["git", "add", "-f", "-A"], cwd=temporary_repo, check=True)
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
                cwd=temporary_repo,
                check=True,
            )
            issue_path.write_text(problem_statement, encoding="utf-8")
            config_path.write_text(yaml.safe_dump(prepare_config(args, model_name), sort_keys=False), encoding="utf-8")

            command = [
                str(SWEAGENT_BIN),
                "run",
                "--config",
                str(config_path),
                "--env.repo.path",
                str(temporary_repo),
                "--problem_statement.path",
                str(issue_path),
                "--output_dir",
                str(instance_dir),
                "--env.deployment.image",
                args.docker_image,
            ]
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["SWE_AGENT_LOG_STREAM_LEVEL"] = "WARNING"
            environment["SWEAGENT_LLM_EVENTS_PATH"] = str(run_dir / f"{run_id}.llm_events.jsonl")
            environment["SWEAGENT_CALL_LIMIT_MARKER_PATH"] = str(instance_dir / "llm_call_limit_reached")
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"RUN {instance_id}: {' '.join(command)}")
            with (run_dir / "run.log").open("w", encoding="utf-8") as run_log:
                process = subprocess.Popen(
                    command,
                    cwd=METHOD_DIR,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    run_log.write(line)
                    run_log.flush()
                    print(f"[{instance_id}] {line}", end="")
                returncode = process.wait()

        if returncode != 0:
            raise RuntimeError(f"SWE-agent exited with code {returncode}")
        completed_run = find_completed_run(instance_dir)
        if completed_run is None:
            raise RuntimeError("SWE-agent exited successfully but did not produce a valid .pred and .traj pair")

        result = {
            "instance_id": instance_id,
            "status": "completed",
            "run_dir": str(completed_run),
            "returncode": returncode,
            "completed_at": utc_now(),
        }
        write_json(status_path, result)
        print(f"DONE {instance_id}: {completed_run}")
        return result
    except Exception as error:
        result = {
            "instance_id": instance_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
        }
        write_json(status_path, result)
        print(f"ERROR {instance_id}: {error}", file=sys.stderr)
        return result


def select_instances(dataset, instance_ids, limit):
    if instance_ids:
        requested = set(instance_ids)
        selected = [instance for instance in dataset if instance.get("instance_id") in requested]
        missing = requested - {instance["instance_id"] for instance in selected}
        if missing:
            raise ValueError(f"Instance IDs not found: {sorted(missing)}")
    else:
        selected = dataset
    if limit is not None:
        selected = selected[:limit]
    ids = [instance["instance_id"] for instance in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate instance_id values")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Running SWE-agent-RepoGraph instances in parallel")
    parser.add_argument("--dataset-json", "--dataset_json", dest="dataset_json", type=Path, default=PROJECT_ROOT / "datasets/AgentIssue-Bench/agent_issue.json")
    parser.add_argument("--repo-base", "--repo_base", dest="repo_base", type=Path, default=PROJECT_ROOT / "datasets/AgentIssue-Bench/repo")
    parser.add_argument("--output-base", "--output_base", dest="output_base", type=Path, default=None)
    parser.add_argument("--instance-id", action="append", dest="instance_ids", help="Only run the specified instance and can be passed in repeatedly")
    parser.add_argument("--limit", type=int, help="Run only the first N instances in the filtered results")
    parser.add_argument("--llm-model", "--model", dest="model", required=True)
    parser.add_argument("--llm-base-url", "--llm_base_url", "--api-base", "--api_base", dest="api_base", required=True)
    parser.add_argument("--llm-api-key", "--llm_api_key", "--api-key", "--api_key", dest="api_key", required=True)
    parser.add_argument("--base-config", "--base_config", dest="base_config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--docker-image", default="python:3.11")
    parser.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int, default=3)
    parser.add_argument("--max-llm-calls", "--max_llm_calls", dest="max_llm_calls", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing output of the selected instance and rerun")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.api_base.startswith(("http://", "https://")):
        parser.error("--api-base must start with http:// or https://")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    args.dataset_json = args.dataset_json.expanduser().resolve()
    args.repo_base = args.repo_base.expanduser().resolve()
    args.base_config = args.base_config.expanduser().resolve()
    if args.output_base is not None:
        args.output_base = args.output_base.expanduser().resolve()
    if not args.dataset_json.is_file():
        parser.error(f"Dataset not found: {args.dataset_json}")
    if not args.repo_base.is_dir():
        parser.error(f"Repository base not found: {args.repo_base}")
    if not args.base_config.is_file():
        parser.error(f"Base config not found: {args.base_config}")
    if not SWEAGENT_BIN.is_file():
        parser.error(f"SWE-agent executable not found: {SWEAGENT_BIN}")

    if args.output_base is None:
        dataset_name = args.dataset_json.parent.name
        model_dir = normalize_model_name(args.model, args.api_base).replace("/", "-")
        args.output_base = PROJECT_ROOT / "results/results_sweagent_repograph" / dataset_name / model_dir
    return args


def main():
    args = parse_args()
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    instances = select_instances(dataset, args.instance_ids, args.limit)
    if not instances:
        raise ValueError("No instances selected")

    print(f"Selected {len(instances)} instances; first two: {[item['instance_id'] for item in instances[:2]]}")
    results = []
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(instances))) as executor:
        future_to_id = {executor.submit(run_instance, instance, args): instance["instance_id"] for instance in instances}
        for future in as_completed(future_to_id):
            instance_id = future_to_id[future]
            try:
                results.append(future.result())
            except Exception as error:
                print(f"ERROR {instance_id}: worker crashed: {error}", file=sys.stderr)
                results.append({"instance_id": instance_id, "status": "failed", "error": str(error)})

    counts = {status: sum(result["status"] == status for result in results) for status in sorted({result["status"] for result in results})}
    print(f"Summary: {counts}")
    if any(result["status"] in {"failed", "blocked", "call_limit"} for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


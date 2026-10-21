import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "LocAgent"
PYTHON_BIN = METHOD_DIR / ".venv/bin/python"


def load_instance_ids(path):
    if not path.is_file():
        return set()
    try:
        with path.open(encoding="utf-8") as file:
            return {
                json.loads(line)["instance_id"]
                for line in file
                if line.strip()
            }
    except (OSError, json.JSONDecodeError, KeyError) as error:
        print(f"Failed to read positioning result: {path}: {error}", file=sys.stderr)
        return set()


def load_call_limited_ids(path):
    if not path.is_file():
        return set()
    try:
        terminal_ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = record["instance_id"]
            if record.get("call_limit_reached"):
                terminal_ids.add(instance_id)
            else:
                terminal_ids.discard(instance_id)
        return terminal_ids
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise RuntimeError(f"Failed to read call upper limit status: {path}: {error}") from error


def parse_args():
    parser = argparse.ArgumentParser(description="Running LocAgent positioning in parallel")
    parser.add_argument("--dataset-json", "--dataset_json", dest="dataset_json", type=Path, default=PROJECT_ROOT / "datasets/AgentIssue-Bench/agent_issue.json")
    parser.add_argument("--repo-base", "--repo_base", dest="repo_base", type=Path, default=PROJECT_ROOT / "datasets/AgentIssue-Bench/repo")
    parser.add_argument("--output-base", "--output_base", dest="output_base", type=Path, default=PROJECT_ROOT / "results/results_locagent/AgentIssue-Bench")
    parser.add_argument("--index-base", "--index_base", dest="index_base", type=Path, default=PROJECT_ROOT / "results/index/LocAgent/AgentIssue-Bench")
    parser.add_argument("--instance-id", action="append", dest="instance_ids", help="Only run the specified instance and can be passed in repeatedly")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="openai/gpt-4o-2024-05-13")
    parser.add_argument("--openai-base-url", "--openai_base_url", dest="openai_base_url", default="http://127.0.0.1:8795/v1")
    parser.add_argument("--openai-api-key", "--openai_api_key", dest="openai_api_key", default="token-123")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=2)
    parser.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int, default=2)
    parser.add_argument("--max-llm-calls", "--max_llm_calls", dest="max_llm_calls", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing output of the selected instance and rerun")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    args.dataset_json = args.dataset_json.expanduser().resolve()
    args.repo_base = args.repo_base.expanduser().resolve()
    args.output_base = args.output_base.expanduser().resolve()
    args.index_base = args.index_base.expanduser().resolve()
    return args


def select_instances(dataset, instance_ids, limit):
    if instance_ids:
        requested = set(instance_ids)
        selected = [item for item in dataset if item.get("instance_id") in requested]
        missing = requested - {item["instance_id"] for item in selected}
        if missing:
            raise ValueError(f"Not found instances: {sorted(missing)}")
    else:
        selected = dataset
    return selected[:limit] if limit is not None else selected


def remove_selected_records(path, instance_ids):
    if not path.is_file():
        return
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Failed to read existing results: {path}: {error}") from error
    remaining = [record for record in records if record.get("instance_id") not in instance_ids]
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in remaining),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    instances = select_instances(dataset, args.instance_ids, args.limit)
    if not instances:
        raise ValueError("No instance selected")
    missing_repos = [item["instance_id"] for item in instances if not (args.repo_base / item["instance_id"]).is_dir()]
    if missing_repos:
        raise FileNotFoundError(f"The warehouse does not exist: {missing_repos}")

    selected_ids = {item["instance_id"] for item in instances}
    raw_file = args.output_base / "loc_outputs.jsonl"
    merged_file = args.output_base / "merged_loc_outputs_mrr.jsonl"

    terminal_ids = set() if args.overwrite else load_call_limited_ids(raw_file) & selected_ids
    if terminal_ids:
        print(f"The LLM call limit has been reached and will not be re-executed: {sorted(terminal_ids)}", file=sys.stderr)
        instances = [item for item in instances if item["instance_id"] not in terminal_ids]
    if not instances:
        return 1
    expected_ids = {item["instance_id"] for item in instances}

    args.output_base.mkdir(parents=True, exist_ok=True)
    args.index_base.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("loc_outputs.jsonl", "loc_trajs.jsonl", "merged_loc_outputs_mrr.jsonl"):
            remove_selected_records(args.output_base / name, expected_ids)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as file:
        json.dump(instances, file, ensure_ascii=False)
        selected_dataset = Path(file.name)

    command = [
        str(PYTHON_BIN if PYTHON_BIN.is_file() else Path(sys.executable)),
        "auto_search_main.py",
        "--dataset_json",
        str(selected_dataset),
        "--model",
        args.model,
        "--localize",
        "--merge",
        "--output_folder",
        str(args.output_base),
        "--num_processes",
        str(min(args.max_workers, len(instances))),
        "--num_samples",
        str(args.num_samples),
        "--max_llm_calls",
        str(args.max_llm_calls),
        "--use_function_calling",
        "--simple_desc",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{METHOD_DIR}:{environment.get('PYTHONPATH', '')}",
            "OPENAI_API_BASE": args.openai_base_url,
            "OPENAI_BASE_URL": args.openai_base_url,
            "OPENAI_API_KEY": args.openai_api_key,
            "REPO_BASE_PATH": str(args.repo_base),
            "GRAPH_INDEX_DIR": str(args.index_base / "graph"),
            "BM25_INDEX_DIR": str(args.index_base / "bm25"),
        }
    )
    print(f"Select {len(instances)} instances, the first two: {[item['instance_id'] for item in instances[:2]]}")
    try:
        process = subprocess.run(command, cwd=METHOD_DIR, env=environment)
    finally:
        selected_dataset.unlink(missing_ok=True)

    completed_ids = load_instance_ids(merged_file)
    missing_ids = expected_ids - completed_ids
    if process.returncode != 0:
        print(f"LocAgent failed, exit code: {process.returncode}", file=sys.stderr)
        return process.returncode
    if missing_ids:
        print(f"Missing final location result: {sorted(missing_ids)}", file=sys.stderr)
        return 1
    print(f"Final completion: {len(expected_ids)}/{len(expected_ids)}")
    return 1 if terminal_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())

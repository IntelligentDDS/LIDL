import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "LocAgent"
SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_locagent_single.py"

DATASETS = [
    {
        "name": "Hydrangea",
        "dataset_json": PROJECT_ROOT / "datasets/Hydrangea/hydrangea_defect.json",
        "repo_base": PROJECT_ROOT / "datasets/Hydrangea/repo",
    },
    {
        "name": "AgentIssue-Bench",
        "dataset_json": PROJECT_ROOT / "datasets/AgentIssue-Bench/agent_issue.json",
        "repo_base": PROJECT_ROOT / "datasets/AgentIssue-Bench/repo",
    },
]

models = [
    {
        "model": "openai/gemma-4-31b",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
    {
        "model": "openai/kimi-k2.5",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
    {
        "model": "openai/glm-5-2-260617",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
    {
        "model": "openai/doubao-seed-2-1-turbo-260628",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 4,
    },
    {
        "model": "openai/deepseek-v4-flash-ga-260731",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 4,
    },
    {
        "model": "openai/gpt-5.6-terra",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
]


def run(dataset, model, overwrite, max_llm_calls):
    model_name = model["model"].split("/", 1)[-1]
    output_base = PROJECT_ROOT / "results/results_locagent" / dataset["name"] / model_name
    index_base = PROJECT_ROOT / "results/index/LocAgent" / dataset["name"] / model_name
    dataset_workers = max(1, model["max_workers"] // len(DATASETS))
    command = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--dataset_json",
        str(dataset["dataset_json"]),
        "--repo_base",
        str(dataset["repo_base"]),
        "--output_base",
        str(output_base),
        "--index_base",
        str(index_base),
        "--model",
        model["model"],
        "--openai_base_url",
        model["base_url"],
        "--openai_api_key",
        model["api_key"],
        "--max_workers",
        str(dataset_workers),
        "--max-llm-calls",
        str(max_llm_calls),
    ]
    if overwrite:
        command.append("--overwrite")
    print(
        f"RUN [{dataset['name']}|{model_name}]: instances=all, "
        f"dataset_workers={dataset_workers}, model_workers={model['max_workers']}"
    )
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=METHOD_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    for line in process.stdout:
        print(f"[{dataset['name']}|{model_name}] {line}", end="")
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run all LocAgent location tasks in parallel")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing results for the selected instance and rerun")
    parser.add_argument("--max-llm-calls", type=int, default=50, help="Maximum number of LLM requests allowed per instance")
    args = parser.parse_args()
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    failed = False
    jobs = [(dataset, model) for dataset in DATASETS for model in models]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run, dataset, model, args.overwrite, args.max_llm_calls): (dataset, model) for dataset, model in jobs}
        for future in as_completed(futures):
            dataset, model = futures[future]
            try:
                code = future.result()
            except Exception as error:
                failed = True
                print(f"Task exception [{dataset['name']}|{model['model']}]: {error}")
                continue
            if code != 0:
                failed = True
                print(f"Task failed [{dataset['name']}|{model['model']}], exit code: {code}")
            else:
                print(f"Mission successful [{dataset['name']}|{model['model']}]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

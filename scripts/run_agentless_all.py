import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = dotenv_values(PROJECT_ROOT / "scripts/.env")
METHOD_DIR = PROJECT_ROOT / "compare_code" / "Agentless"
SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_agentless_single.py"
EMBEDDING_MODEL = ENVIRONMENT.get("EMBEDDING_MODEL")
EMBEDDING_BASE_URL = ENVIRONMENT.get("EMBEDDING_BASE_URL")
EMBEDDING_API_KEY = ENVIRONMENT.get("EMBEDDING_API_KEY")

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
    output_base = PROJECT_ROOT / "results/results_agentless" / dataset["name"] / model_name
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
        "--backend",
        "openai",
        "--model",
        model["model"],
        "--openai_base_url",
        model["base_url"],
        "--openai_api_key",
        model["api_key"],
        "--embedding_type",
        "openai",
        "--embedding_model",
        EMBEDDING_MODEL,
        "--embedding_base_url",
        EMBEDDING_BASE_URL,
        "--embedding_api_key",
        EMBEDDING_API_KEY,
        "--embedding_dimensions",
        "1024",
        "--max_workers",
        str(dataset_workers),
        "--max-llm-calls",
        str(max_llm_calls),
    ]
    if overwrite:
        command.append("--overwrite")
    print(
        f"RUN [{dataset['name']}|{model['model']}]: instances=all, "
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
        print(f"[{dataset['name']}|{model['model']}] {line}", end="")
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run all Agentless positioning tasks in parallel")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing results and rerun")
    parser.add_argument("--max-llm-calls", type=int, default=50, help="Maximum number of LLM requests allowed per instance")
    args = parser.parse_args()
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    missing = [name for name, value in (
        ("EMBEDDING_MODEL", EMBEDDING_MODEL),
        ("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL),
        ("EMBEDDING_API_KEY", EMBEDDING_API_KEY),
    ) if not value]
    if missing:
        parser.error(f"Missing environment variable: {', '.join(missing)}")

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

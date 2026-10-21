import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_codex_single.py"
DATASETS = [
    {
        "name": "Hydrangea",
        "dataset_json": PROJECT_ROOT / "datasets" / "Hydrangea" / "hydrangea_defect.json",
        "repo_base": PROJECT_ROOT / "datasets" / "Hydrangea" / "repo",
    },
    {
        "name": "AgentIssue-Bench",
        "dataset_json": PROJECT_ROOT / "datasets" / "AgentIssue-Bench" / "agent_issue.json",
        "repo_base": PROJECT_ROOT / "datasets" / "AgentIssue-Bench" / "repo",
    },
]
MODELS = [
    {
        "model": "gemma-4-31b",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
    {
        "model": "kimi-k2.5",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
    {
        "model": "glm-5-2-260617",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
    {
        "model": "doubao-seed-2-1-turbo-260628",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 4,
    },
    {
        "model": "deepseek-v4-flash-ga-260731",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 4,
    },
    {
        "model": "gpt-5.6-terra",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": "sk-1234",
        "max_workers": 2,
    },
]


def run(dataset, model, args):
    model_name = model["model"]
    output_base = args.output_root / dataset["name"] / model_name
    dataset_workers = model["max_workers"]
    command = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--dataset-json",
        str(dataset["dataset_json"]),
        "--repo-base",
        str(dataset["repo_base"]),
        "--output-base",
        str(output_base),
        "--max-workers",
        str(dataset_workers),
        "--model",
        model_name,
        "--model-provider",
        args.model_provider,
        "--base-url",
        model["base_url"],
        "--api-key",
        model["api_key"],
        "--api-key-env",
        args.api_key_env,
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in process.stdout:
        print(f"[{dataset['name']}|{model_name}] {line}", end="", flush=True)
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run Codex SDK across all datasets and models")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "results_codex")
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if any(model["max_workers"] < 1 for model in MODELS):
        parser.error("Every model max_workers must be at least 1")
    args.output_root = args.output_root.expanduser().resolve()
    jobs = [(dataset, model) for dataset in DATASETS for model in MODELS]
    logger.info(f"Starting {len(jobs)} Codex jobs; first two: {jobs[:2]}")
    failed = False
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run, dataset, model, args): (dataset, model) for dataset, model in jobs}
        for future in as_completed(futures):
            dataset, model = futures[future]
            model_name = model["model"]
            try:
                return_code = future.result()
            except Exception as error:
                failed = True
                logger.error(f"Codex job crashed for {dataset['name']} {model_name}: {error}")
                continue
            if return_code != 0:
                failed = True
                logger.error(f"Codex job failed for {dataset['name']} {model_name}: exit code {return_code}")
            else:
                logger.info(f"Codex job completed: {dataset['name']} {model_name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

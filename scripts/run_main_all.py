import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_main_single.py"
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
        "model": "openai/gemma-4-31b",
        "base_url": "http://127.0.0.1:8322/v1",
        "max_workers": 2,
    },
    {
        "model": "openai/kimi-k2.5",
        "base_url": "http://127.0.0.1:8322/v1",
        "max_workers": 2,
    },
    {
        "model": "openai/glm-5-2-260617",
        "base_url": "http://127.0.0.1:8322/v1",
        "max_workers": 2,
    },
    {
        "model": "openai/doubao-seed-2-1-turbo-260628",
        "base_url": "http://127.0.0.1:8322/v1",
        "max_workers": 4,
    },
    {
        "model": "openai/deepseek-v4-flash-ga-260731",
        "base_url": "http://127.0.0.1:8322/v1",
        "max_workers": 4,
    },
    {
        "model": "openai/gpt-5.6-terra",
        "base_url": "http://127.0.0.1:8322/v1",
        "max_workers": 2,
    },
]


def run(dataset, model, args):
    model_name = model["model"].split("/", 1)[-1]
    output_base = args.output_root / dataset["name"] / model_name
    dataset_workers = max(1, model["max_workers"] // len(DATASETS))
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
        "--cache-base",
        str(args.cache_base),
        "--model",
        model["model"],
        "--openai-base-url",
        model["base_url"],
        "--seed-count",
        str(args.seed_count),
        "--candidate-count",
        str(args.candidate_count),
        "--entity-budget",
        str(args.entity_budget),
        "--line-token-budget-ratio",
        str(args.line_token_budget_ratio),
        "--lexical-weight",
        str(args.lexical_weight),
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-output-tokens",
        str(args.max_output_tokens),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.overwrite:
        command.append("--overwrite")
    logger.info(
        f"Starting {dataset['name']} {model_name}: "
        f"dataset_workers={dataset_workers}, model_workers={model['max_workers']}"
    )
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
    load_dotenv(PROJECT_ROOT / "scripts" / ".env")
    parser = argparse.ArgumentParser(description="Run all LIDL models across both datasets")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "results" / "results_main"
    )
    parser.add_argument("--cache-base", type=Path, default=PROJECT_ROOT / "results" / "results_main" / "._cache")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--model",
        action="append",
        choices=[model["model"].split("/", 1)[-1] for model in MODELS],
        help="Run only the selected model; repeat this option to select multiple models",
    )
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--entity-budget", type=int, default=40)
    parser.add_argument("--line-token-budget-ratio", type=float, default=0.9)
    parser.add_argument("--lexical-weight", type=float, default=0.65)
    parser.add_argument("--max-input-tokens", type=int, default=180000)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not os.getenv("LLM_API_KEY"):
        parser.error("Missing LLM_API_KEY in scripts/.env or the environment")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if (
        args.seed_count < 1
        or args.candidate_count < 1
        or args.entity_budget < 1
        or not 0 < args.line_token_budget_ratio <= 1
        or not 0 <= args.lexical_weight <= 1
    ):
        parser.error("Invalid retrieval configuration")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")
    if args.max_input_tokens <= args.max_output_tokens:
        parser.error("--max-input-tokens must exceed --max-output-tokens")
    args.output_root = args.output_root.expanduser().resolve()
    args.cache_base = args.cache_base.expanduser().resolve()

    selected_models = [
        model
        for model in MODELS
        if not args.model or model["model"].split("/", 1)[-1] in args.model
    ]
    failed = False
    jobs = [(dataset, model) for dataset in DATASETS for model in selected_models]
    logger.info(f"Starting {len(jobs)} dataset-model jobs; first two: {jobs[:2]}")
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run, dataset, model, args): (dataset, model) for dataset, model in jobs}
        for future in as_completed(futures):
            dataset, model = futures[future]
            model_name = model["model"].split("/", 1)[-1]
            try:
                return_code = future.result()
            except Exception as error:
                failed = True
                logger.error(f"Job crashed for {dataset['name']} {model_name}: {error}")
                continue
            if return_code != 0:
                failed = True
                logger.error(f"Job failed for {dataset['name']} {model_name}: exit code {return_code}")
            else:
                logger.info(f"Job completed: {dataset['name']} {model_name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

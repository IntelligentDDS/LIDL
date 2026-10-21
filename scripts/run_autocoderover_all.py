import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "auto-code-rover"
SINGLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_autocoderover_single.py"

datasets = [
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
        "api_key": os.getenv("LLM_API_KEY", ""),
        "max_workers": 2,
    },
    {
        "model": "openai/kimi-k2.5",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "max_workers": 2,
    },
    {
        "model": "openai/glm-5-2-260617",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "max_workers": 2,
    },
    {
        "model": "openai/doubao-seed-2-1-turbo-260628",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "max_workers": 4,
    },
    {
        "model": "openai/deepseek-v4-flash-ga-260731",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "max_workers": 4,
    },
    {
        "model": "openai/gpt-5.6-terra",
        "base_url": "http://127.0.0.1:8322/v1",
        "api_key": os.getenv("LLM_API_KEY", ""),
        "max_workers": 2,
    },
    # {
    #     "model": "openai/deepseek-v3-2-251201",
    #     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    #     "api_key": os.getenv("LLM_API_KEY", ""),
    #     "max_workers": 3,
    # },
    # {
    #     "model": "openai/kimi-k2-250905",
    #     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    #     "api_key": os.getenv("LLM_API_KEY", ""),
    #     "max_workers": 3,
    # },
    # {
    #     "model": "openai/gpt-5.1",
    #     "base_url": "https://api.openai.com/v1",
    #     "api_key": os.getenv("LLM_API_KEY", ""),
    #     "max_workers": 3,
    # },
    # {
    #     "model": "openai/claude-sonnet-4-5",
    #     "base_url": "https://api.anthropic.com/v1",
    #     "api_key": os.getenv("LLM_API_KEY", ""),
    #     "max_workers": 3,
    # },
]


def run(dataset, model, overwrite, max_llm_calls):
    model_name = model["model"].split("/")[-1]
    output_dir = PROJECT_ROOT / "results" / "results_autocoderover" / dataset["name"] / model_name
    dataset_workers = max(1, model["max_workers"] // len(datasets))
    command = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--dataset_json",
        str(dataset["dataset_json"]),
        "--repo_base",
        str(dataset["repo_base"]),
        "--output_dir",
        str(output_dir),
        "--model",
        model["model"],
        "--model_temp",
        "0",
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
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=METHOD_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in process.stdout:
        print(f"[{dataset['name']}|{model_name}] {line}", end="")
    return process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run all Auto-Code-Rover positioning tasks in parallel")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing results for the selected instance and rerun")
    parser.add_argument("--max-llm-calls", type=int, default=50, help="Maximum number of LLM requests allowed per instance")
    args = parser.parse_args()
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    failed = False
    jobs = [(dataset, model) for dataset in datasets for model in models]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run, dataset, model, args.overwrite, args.max_llm_calls): (dataset, model) for dataset, model in jobs}
        for future in as_completed(futures):
            dataset, model = futures[future]
            model_name = model["model"].split("/")[-1]
            try:
                code = future.result()
            except Exception as e:
                failed = True
                print(f"Task exception [{dataset['name']}|{model_name}]: {e}")
                continue
            if code != 0:
                failed = True
                print(f"Task failed [{dataset['name']}|{model_name}], exit code: {code}")
            else:
                print(f"Mission successful [{dataset['name']}|{model_name}]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

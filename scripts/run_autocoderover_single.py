import os
import json
import signal
import subprocess
import argparse
from pathlib import Path
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "auto-code-rover"


def has_completed_run(output_dir, instance_id):
    success_message = f"Task {instance_id} completed successfully."
    for info_log in Path(output_dir).glob(f"{instance_id}_*/info.log"):
        try:
            if success_message in info_log.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError as e:
            print(f"Failed to read operation log: {info_log}: {e}")
    return False


def call_limit_reached(output_dir, instance_id, max_llm_calls):
    marker = "Maximum LLM call limit reached"
    for run_dir in Path(output_dir).glob(f"{instance_id}_*"):
        try:
            if (run_dir / "llm_call_limit_reached").is_file():
                return True
            count_path = run_dir / "llm_call_count.txt"
            limit_path = run_dir / "llm_call_limit.txt"
            if not limit_path.is_file() and count_path.is_file() and int(count_path.read_text(encoding="utf-8") or "0") >= max_llm_calls:
                return True
            info_log = run_dir / "info.log"
            if info_log.is_file() and marker in info_log.read_text(encoding="utf-8", errors="ignore"):
                return True
        except (OSError, ValueError) as e:
            print(f"Failed to read call upper limit status: {run_dir}: {e}")
    return False


def run_local_issue(instance, args):
    instance_id = instance["instance_id"]
    source_repo_path = Path(args.repo_base) / instance_id

    if has_completed_run(args.output_dir, instance_id) and not args.overwrite:
        print(f"Instance {instance_id} already processed!")
        return True
    if call_limit_reached(args.output_dir, instance_id, args.max_llm_calls):
        print(f"Instance {instance_id} reached the terminal {args.max_llm_calls}-call limit; not rerunning.")
        return has_completed_run(args.output_dir, instance_id)
    print(f"Instance {instance_id} will be processed...")

    # If there is, remove it
    for d in Path(args.output_dir).glob(f"**/{instance_id}_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    os.makedirs("./temp", exist_ok=True)
    temp_repo_root = tempfile.mkdtemp(prefix=f"acr_{instance_id}_", dir="./temp")
    repo_path = Path(temp_repo_root) / instance_id
    temp_issue_file = None

    try:
        shutil.copytree(
            source_repo_path,
            repo_path,
            ignore=shutil.ignore_patterns(".git"),
        )
        # Only save problem_statement to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir="./temp", delete=False) as f:
            f.write(instance["problem_statement"])
            temp_issue_file = f.name

        command = (
            f"python app/main.py local-issue "
            f"--output-dir {args.output_dir} "
            f"--model {args.model} "
            f"--model-temperature {args.model_temp} "
            f"--max-llm-calls {args.max_llm_calls} "
            f"--local-repo {repo_path} "
            f"--issue-file {temp_issue_file} "
            f"--task-id {instance_id}"
        )

        command_str = f"PYTHONPATH=. {command}"
        print(f"▶ Running: {command_str}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["REPO_BASE_PATH"] = args.repo_base

        process = subprocess.Popen(
            command_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )

        completed_successfully = False
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if f"Task {instance_id} completed successfully." in line:
                    completed_successfully = True
            process.wait()
            completed_successfully = completed_successfully or has_completed_run(
                args.output_dir, instance_id
            )
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise

        if process.returncode != 0 or not completed_successfully:
            print(f"Instance {instance_id} failed!")
            return False

        print(f"Instance {instance_id} completed!")
        return True

    finally:
        if temp_issue_file and os.path.exists(temp_issue_file):
            os.remove(temp_issue_file)
        shutil.rmtree(temp_repo_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run local issue handling")
    parser.add_argument("--dataset_json", default=PROJECT_ROOT / "datasets/AgentIssue-Bench/agent_issue.json", help="Enter JSON file path")
    parser.add_argument("--repo_base", default=PROJECT_ROOT / "datasets/AgentIssue-Bench/repo", help="Warehouse base path")
    parser.add_argument("--output_dir", default=PROJECT_ROOT / "results/results_autocoderover/AgentIssue-Bench", help="Output directory")
    parser.add_argument("--limit", type=int, help="Only run the first N instances")
    parser.add_argument("--model", default="openai/llama-3.3-70b-instruct", help="Model name")
    parser.add_argument("--model_temp", default="0", help="model temperature")
    parser.add_argument("--openai_base_url", default="http://localhost:8795/v1", help="OpenAI Base URL")
    parser.add_argument("--openai_api_key", default="token-123", help="OpenAI API Key")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of instances processed concurrently")
    parser.add_argument("--max-llm-calls", "--max_llm_calls", dest="max_llm_calls", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing output of the selected instance and rerun")

    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be at least 1")
    args.dataset_json = Path(args.dataset_json).expanduser().resolve()
    args.repo_base = Path(args.repo_base).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    os.chdir(METHOD_DIR)

    # Set environment variables
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    os.environ["OPENAI_API_BASE_URL"] = args.openai_base_url
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["OPENAI_KEY"] = args.openai_api_key

    # Read the data set
    with open(args.dataset_json) as f:
        dataset = json.load(f)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    worker_count = min(args.max_workers, len(dataset))
    print(f"The data set contains {len(dataset)} instances, using {worker_count} concurrent processes")

    # Process each instance
    success_count = 0
    if worker_count == 1:
        for i, instance in enumerate(dataset, 1):
            if run_local_issue(instance, args):
                success_count += 1
            print(f"Progress: {i}/{len(dataset)}, Success: {success_count}")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_instance = {executor.submit(run_local_issue, instance, args): instance for instance in dataset}
            for i, future in enumerate(as_completed(future_to_instance), 1):
                instance = future_to_instance[future]
                try:
                    if future.result():
                        success_count += 1
                    print(f"Progress: {i}/{len(dataset)}, Success: {success_count}")
                except Exception as e:
                    print(f"Instance {instance['instance_id']} Exception occurred: {e}")
    print(f"\n final completion: {success_count}/{len(dataset)}")
    return 0 if success_count == len(dataset) else 1
    # for i, instance in enumerate(dataset):
    #     print(f"\n{'='*60}\n processing Instance {i+1}: {instance['instance_id']}\n{'='*60}")
    #     if run_local_issue(instance, args):
    #         success_count += 1
    #     print(f"Complete: {success_count}/{len(dataset)}")

if __name__ == "__main__":
    raise SystemExit(main())


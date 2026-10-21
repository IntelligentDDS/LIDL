import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import fcntl
from concurrent.futures._base import Future
import json
import shutil
import subprocess
import sys
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = PROJECT_ROOT / "compare_code" / "Agentless-RepoGraph"
MAX_LOCATIONS = 10


def has_final_record(path, instance_id):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as file:
            return any(
                record.get("instance_id") == instance_id
                and bool(record.get("found_edit_locs"))
                for line in file
                if line.strip()
                for record in [json.loads(line)]
            )
    except (OSError, json.JSONDecodeError) as error:
        print(f"Failed to read final result: {path}: {error}")
        return False


def read_call_limit(output_base, max_llm_calls):
    count_path = Path(output_base) / "llm_call_count.txt"
    limit_path = Path(output_base) / "llm_call_limit.txt"
    marker_path = Path(output_base) / "llm_call_limit_reached"
    try:
        count = int(count_path.read_text(encoding="utf-8") or "0") if count_path.is_file() else 0
        limit = int(limit_path.read_text(encoding="utf-8")) if limit_path.is_file() else 50
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to read LLM call cap status: {output_base}: {error}") from error
    reached = marker_path.is_file() or (not limit_path.is_file() and count >= limit)
    return reached, limit


def ensure_repo_graph(instance_id, repo_path):
    graph_dir = METHOD_DIR / "repo_structures" / "graph"
    graph_path = graph_dir / f"{instance_id}.pkl"
    tags_path = graph_dir / f"tags_{instance_id}.json"
    graph_dir.mkdir(parents=True, exist_ok=True)

    lock_path = graph_dir / f"{instance_id}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if graph_path.is_file() and tags_path.is_file():
            return

        graph_python = METHOD_DIR.parent / "SWE-agent-RepoGraph" / ".venv" / "bin" / "python"
        if not graph_python.is_file():
            graph_python = Path(sys.executable)
        command = [
            str(graph_python),
            str(METHOD_DIR / "repograph" / "construct_graph.py"),
            str(repo_path),
        ]
        print(f"Build RepoGraph: {' '.join(command)}")
        with tempfile.TemporaryDirectory(prefix=f"repograph-{instance_id}-") as temporary_dir:
            temporary_dir = Path(temporary_dir)
            subprocess.run(command, cwd=temporary_dir, check=True)
            built_graph = temporary_dir / "graph.pkl"
            built_tags = temporary_dir / "tags.json"
            if not built_graph.is_file() or not built_tags.is_file():
                raise RuntimeError(f"RepoGraph did not generate graph file: {instance_id}")
            shutil.move(built_graph, graph_path)
            tags = [json.loads(line) for line in built_tags.read_text().splitlines() if line]
            tags_path.write_text(json.dumps(tags))


def run_instance_pipeline(instance, args):
    """Run a complete pipeline for a single instance"""
    instance_id = instance["instance_id"]

    # Create temporary JSON file
    os.makedirs("./temp", exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir="./temp", delete=False) as f:
        json.dump([instance], f)
        temp_json = f.name

    # Define output path
    output_base = f"{args.output_base}/{instance_id}"
    embedding_persist_base = f"{args.embedding_persist_base}/{instance_id}"
    try:
        final_file = os.path.join(output_base, "edit_location_individual", "loc_merged_0-0_outputs.jsonl")
        call_limit_reached, call_limit = read_call_limit(output_base, args.max_llm_calls)
        if call_limit_reached:
            print(f"Instance {instance_id} has reached the upper limit of {call_limit} LLM calls and will not be re-executed.")
            return has_final_record(final_file, instance_id)
        if has_final_record(final_file, instance_id) and not args.overwrite:
            print(f"Already exists: {final_file}, skip {instance_id}")
            return True
        if args.overwrite and os.path.exists(output_base):
            print(f"Overwrite existing results: {output_base}")
            shutil.rmtree(output_base)
        os.makedirs(output_base, exist_ok=True)
        call_count_file = os.path.join(output_base, "llm_call_count.txt")
        if not os.path.exists(call_count_file):
            Path(call_count_file).write_text("0", encoding="utf-8")
        Path(output_base, "llm_call_limit.txt").write_text(str(args.max_llm_calls), encoding="utf-8")

        repo_path = Path(args.repo_base) / instance_id
        if not repo_path.is_dir():
            raise FileNotFoundError(f"The warehouse does not exist: {repo_path}")
        ensure_repo_graph(instance_id, repo_path)

        commands = [
            # Step 1: File level
            f"python agentless/fl/localize.py --file_level "
            f"--output_folder {output_base}/file_level "
            f"--num_threads 1 --skip_existing "
            f"--backend {args.backend} --model {args.model} "
            f"--dataset_json {temp_json}",
            # Step 2: Irrelevant
            f"python agentless/fl/localize.py --file_level --irrelevant "
            f"--output_folder {output_base}/file_level_irrelevant "
            f"--num_threads 1 --skip_existing "
            f"--backend {args.backend} --model {args.model} "
            f"--dataset_json {temp_json}",
            # Step 3: Retrieve
            f"python agentless/fl/retrieve.py --index_type simple --filter_type given_files "
            f"--filter_file {output_base}/file_level_irrelevant/loc_outputs.jsonl "
            f"--output_folder {output_base}/retrievel_embedding "
            f"--persist_dir {embedding_persist_base} "
            f"--num_threads 1 --dataset_json {temp_json} "
            f"--embedding_type {args.embedding_type} "
            f"--embedding_model {args.embedding_model} "
            f"--embedding_base_url {args.embedding_base_url} "
            f"--embedding_api_key {args.embedding_api_key} "
            f"--embedding_dimensions {args.embedding_dimensions}",
            # Step 4: Combine. before calculating this step.
            f"python agentless/fl/combine.py "
            f"--retrieval_loc_file {output_base}/retrievel_embedding/retrieve_locs.jsonl "
            f"--model_loc_file {output_base}/file_level/loc_outputs.jsonl "
            f"--top_n {MAX_LOCATIONS} --output_folder {output_base}/file_level_combined",
            # Step 5: Related elements
            f"python agentless/fl/localize.py --related_level "
            f"--output_folder {output_base}/related_elements "
            f"--top_n {MAX_LOCATIONS} --compress_assign --compress "
            f"--start_file {output_base}/file_level_combined/combined_locs.jsonl "
            f"--num_threads 1 --skip_existing "
            f"--backend {args.backend} --model {args.model} "
            f"--dataset_json {temp_json}",
            # Step 6: Fine grain
            f"python agentless/fl/localize.py --fine_grain_line_level "
            f"--output_folder {output_base}/edit_location_samples "
            f"--top_n {MAX_LOCATIONS} --compress --temperature 0 --num_samples 1 "
            f"--start_file {output_base}/related_elements/loc_outputs.jsonl "
            f"--num_threads 1 --skip_existing "
            f"--backend {args.backend} --model {args.model} "
            f"--dataset_json {temp_json} "
            f"--repo_graph",
            # Step 7: Merge
            f"python agentless/fl/localize.py --merge "
            f"--output_folder {output_base}/edit_location_individual "
            f"--top_n {MAX_LOCATIONS} --num_samples 1 "
            f"--start_file {output_base}/edit_location_samples/loc_outputs.jsonl "
            f"--backend {args.backend} --model {args.model} "
            f"--dataset_json {temp_json}",
        ]

        for i, cmd in enumerate(commands, 1):
            print(f"\n steps {i}/ {len(commands)}...")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["REPO_BASE_PATH"] = args.repo_base
            env["LLM_CALL_COUNT_FILE"] = call_count_file
            env["MAX_LLM_CALLS"] = str(args.max_llm_calls)
            print(cmd)
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
            )
            for line in process.stdout:
                print(line, end="", flush=True)
            process.wait()
            if process.returncode != 0:
                print(f"Step {i} failed with exit code: {process.returncode}")
                return False

        if not has_final_record(final_file, instance_id):
            print(f"Instance {instance_id} No final record generated: {final_file}")
            return False
        print(f"Instance {instance_id} Complete")
        return True

    finally:
        if os.path.exists(temp_json):
            os.remove(temp_json)


def main():
    parser = argparse.ArgumentParser(description="Run the Pipeline one instance at a time")
    parser.add_argument(
        "--dataset_json", default=PROJECT_ROOT / "datasets/AgentIssue-Bench/agent_issue.json", help="Enter JSON file path"
    )
    parser.add_argument("--instance-id", dest="instance_id", help="Only run specified instance")
    parser.add_argument("--limit", type=int, help="Only run the first N instances in the filtered results")
    parser.add_argument("--repo_base", default=PROJECT_ROOT / "datasets/AgentIssue-Bench/repo", help="Warehouse base path")
    parser.add_argument("--output_base", default=PROJECT_ROOT / "results/results_agentless_repograph/AgentIssue-Bench", help="Output base path")
    parser.add_argument(
        "--embedding_persist_base",
        default=PROJECT_ROOT / "results/embedding/Agentless-RepoGraph/AgentIssue-Bench",
        help="Embedding base path",
    )
    parser.add_argument("--backend", default="openai", help="backend type")
    parser.add_argument("--model", default="llama-3.3-70b-instruct", help="Model name")
    parser.add_argument("--openai_base_url", default="http://localhost:8795/v1", help="OpenAI Base URL")
    parser.add_argument("--openai_api_key", default="token-123", help="OpenAI API Key")
    parser.add_argument(
        "--embedding_type", type=str, default="ollama", choices=["openai", "ollama"], help="Embedding type"
    )
    parser.add_argument("--embedding_model", default="bge-m3:567m-fp16", help="Embedding model name")
    parser.add_argument("--embedding_base_url", default="http://localhost:11434/v1", help="Embedding Base URL")
    parser.add_argument("--embedding_api_key", default="ollama", help="Embedding API Key")
    parser.add_argument("--embedding_dimensions", type=int, default=1024, help="Embedding dimensions")
    parser.add_argument("--max_workers", type=int, default=3, help="Number of instances processed concurrently")
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
    args.output_base = Path(args.output_base).expanduser().resolve()
    args.embedding_persist_base = Path(args.embedding_persist_base).expanduser().resolve()
    os.chdir(METHOD_DIR)
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{os.getcwd()}"
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Read the data set
    with open(args.dataset_json, "r") as f:
        dataset = json.load(f)
    if args.instance_id:
        dataset = [instance for instance in dataset if instance.get("instance_id") == args.instance_id]
        if not dataset:
            parser.error(f"Instance not found: {args.instance_id}")
    if args.limit is not None:
        dataset = dataset[: args.limit]

    worker_count = min(args.max_workers, len(dataset))
    print(f"The data set contains {len(dataset)} instances, using {worker_count} concurrent processes")

    success_count = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_instance = {executor.submit(run_instance_pipeline, instance, args): instance for instance in dataset}
        for i, future in enumerate[Future[bool]](as_completed(future_to_instance), 1):
            instance = future_to_instance[future]
            try:
                if future.result():
                    success_count += 1
                print(f"Progress: {i}/{len(dataset)}, Success: {success_count}")
            except Exception as e:
                print(f"Instance {instance['instance_id']} Exception occurred: {e}")
    print(f"\n final completion: {success_count}/{len(dataset)}")
    return 0 if success_count == len(dataset) else 1

    # # Process each instance
    # success_count = 0
    # for i, instance in enumerate(dataset):
    #     print(f"\n{'='*60}\n processing Instance {i+1}: {instance['instance_id']}\n{'='*60}")
    #     if run_instance_pipeline(instance, args):
    #         success_count += 1
    #     print(f"Complete: {success_count}/{len(dataset)}")


if __name__ == "__main__":
    raise SystemExit(main())


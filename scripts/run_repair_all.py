import argparse
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from dotenv import load_dotenv
from run_repair_single import (
    PROJECT_ROOT, DATASETS, APPROACH_ROOTS, read_json, infer_autocoderover_instance, build_parser, detect_result,
    infer_instance_id, output_dir_for, process_job, latest_trajectory,
)


def default_result_paths(localization_model):
    paths = []
    for dataset in DATASETS:
        for approach, root_name in APPROACH_ROOTS.items():
            model_names = (
                [f"openai-{localization_model}", localization_model]
                if approach.startswith("SWE-agent")
                else [localization_model]
            )
            path = next(
                (
                    PROJECT_ROOT / "results" / root_name / dataset["name"] / model_name
                    for model_name in model_names
                    if (
                        PROJECT_ROOT / "results" / root_name / dataset["name"] / model_name
                    ).is_dir()
                ),
                None,
            )
            if path is not None:
                paths.append(path)
            elif approach == "RepoGraph":
                logger.warning(
                    f"Optional RepoGraph localization result directory does not exist for "
                    f"{dataset['name']} {localization_model}"
                )
            else:
                logger.warning(
                    f"Localization result directory does not exist for "
                    f"{approach} {dataset['name']} {localization_model}"
                )
    return paths


def is_instance_result(result_path, approach):
    if approach.startswith("Agentless"):
        return (result_path / "edit_location_individual/loc_merged_0-0_outputs.jsonl").exists()
    if approach == "AutoCodeRover":
        return (result_path / "meta.json").exists() or any(result_path.glob("output_*"))
    if approach == "LocAgent":
        return False
    if approach in {"Codex", "LIDL", "RepoGraph"}:
        return (result_path / "output.json").is_file()
    return any(result_path.glob("*.traj")) or any(result_path.glob("*/*.traj"))


def discover_jobs(result_path, instance_ids, limit):
    result_path = result_path.expanduser().resolve()
    approach, _, _ = detect_result(result_path)
    if approach == "LocAgent":
        source_path = result_path / "merged_loc_outputs_mrr.jsonl"
        records = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        available_ids = [record.get("instance_id") for record in records]
        selected_ids = (
            [instance_id for instance_id in instance_ids if instance_id in available_ids]
            if instance_ids
            else [instance_id for instance_id in available_ids if instance_id]
        )
        return [(result_path, instance_id) for instance_id in selected_ids[:limit]]
    candidates = [path for path in sorted(result_path.iterdir()) if path.is_dir()]
    if approach.startswith("SWE-agent"):
        candidates = [path for path in candidates if latest_trajectory(path) is not None]
    if instance_ids:
        by_instance = {infer_instance_id(path, approach, None): path for path in candidates}
        candidates = [by_instance[instance_id] for instance_id in instance_ids if instance_id in by_instance]
    return [(path, None) for path in candidates[:limit]]


def main():
    load_dotenv(PROJECT_ROOT / "scripts/.env")
    parser = build_parser()
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    args.api_key = os.getenv("LLM_API_KEY") or args.api_key
    if args.model != "gpt-5.6-terra" or args.localization_model != "gpt-5.6-terra":
        parser.error("repair and localization models must both be gpt-5.6-terra")
    if args.max_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("max workers and limit must be positive")
    if args.repo_path and len(args.result_paths) != 1:
        parser.error("--repo-path supports only one directly specified result path")
    if not args.prompt_only and not args.api_key:
        parser.error("--api-key or OPENAI_API_KEY is required")

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    try:
        result_paths = args.result_paths or default_result_paths(args.localization_model)
        if not result_paths:
            raise ValueError("No standard localization result directories were found")
        jobs = []
        for result_path in result_paths:
            resolved_path = result_path.expanduser().resolve()
            approach, _, localization_model = detect_result(resolved_path)
            if localization_model != args.localization_model:
                raise ValueError(
                    f"Localization model must be {args.localization_model}, got "
                    f"{localization_model}: {resolved_path}"
                )
            if is_instance_result(resolved_path, approach):
                requested_id = args.instance_ids[0] if args.instance_ids else None
                jobs.append((resolved_path, requested_id))
            else:
                jobs.extend(discover_jobs(resolved_path, args.instance_ids, args.limit or sys.maxsize))
        unique_jobs = list(dict.fromkeys(jobs))
        if not unique_jobs:
            raise ValueError("No localization result jobs were found")
        batch_mode = len(unique_jobs) > 1
        jobs_by_output = {}
        for path, instance_id in unique_jobs:
            approach, dataset, model = detect_result(path)
            localization = {"dataset": dataset, "model": model, "approach": approach,
                            "instance_id": infer_instance_id(path, approach, instance_id)}
            output_dir = output_dir_for(localization, args, batch_mode)
            previous = jobs_by_output.get(output_dir)
            if previous and previous != (path, instance_id):
                raise ValueError(
                    f"Multiple localization jobs target the same output directory {output_dir}: "
                    f"{previous} and {(path, instance_id)}"
                )
            jobs_by_output[output_dir] = (path, instance_id)
        unique_jobs = list(jobs_by_output.values())
        failures = {}
        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(unique_jobs))) as executor:
            futures = {
                executor.submit(process_job, path, instance_id, args, batch_mode): (path, instance_id)
                for path, instance_id in unique_jobs
            }
            for future in as_completed(futures):
                path, instance_id = futures[future]
                try:
                    completed_id, generated = future.result()
                    if generated:
                        logger.info(f"Completed repair generation: {completed_id}")
                    else:
                        failures[f"{path}:{completed_id}"] = "No patch generated"
                        logger.warning(f"Repair generation produced no patch: {completed_id}")
                except Exception as error:
                    key = instance_id or str(path)
                    failures[key] = str(error)
                    logger.error(f"Repair generation failed for {key}: {error}")
        if failures:
            logger.error(f"Failed jobs: {failures}")
            return 1
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        logger.error(f"Repair runner failed: {error}")
        return 1



if __name__ == "__main__":
    raise SystemExit(main())

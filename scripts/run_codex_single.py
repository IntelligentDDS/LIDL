import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from openai_codex import Codex, CodexConfig, Sandbox
from openai_codex.generated.v2_all import TurnStatus

from utils import _parse_json, atomic_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_locations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["path", "line", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranked_locations"],
    "additionalProperties": False,
}


def selected_instances(dataset, instance_ids, limit):
    if instance_ids:
        requested = set(instance_ids)
        selected = [item for item in dataset if item["instance_id"] in requested]
        missing = requested - {item["instance_id"] for item in selected}
        if missing:
            raise ValueError(f"Instances not found: {sorted(missing)}")
    else:
        selected = dataset
    return selected[:limit] if limit is not None else selected


def resolve_location_path(repo_path, raw_path):
    path = raw_path.removeprefix("./")
    if Path(path).is_absolute():
        return None
    if (repo_path / path).is_file():
        return path
    matches = [candidate for candidate in repo_path.rglob(Path(path).name) if candidate.is_file()]
    suffix_matches = [candidate for candidate in matches if candidate.as_posix().endswith(f"/{path}")]
    candidates = suffix_matches or matches
    if len(candidates) != 1:
        return None
    return candidates[0].relative_to(repo_path).as_posix()


def parse_response(content):
    try:
        response = _parse_json(content)
    except ValueError as json_error:
        response = None
    else:
        json_error = ValueError("Codex response must contain a ranked_locations array")
        if isinstance(response, dict) and isinstance(response.get("ranked_locations"), list):
            return response
    locations = []
    pattern = re.compile(r"`?([^`\s:]+\.[A-Za-z0-9]+):(\d+)`?")
    for path, line in pattern.findall(content):
        locations.append(
            {
                "path": path.lstrip("-0123456789. "),
                "line": int(line),
                "reason": "Codex returned this location in an unstructured response.",
            }
        )
    if not locations:
        raise json_error
    logger.warning(f"Codex returned unstructured locations: {locations[:2]}")
    return {"ranked_locations": locations}


def run_instance(instance, args):
    instance_id = instance["instance_id"]
    repo_path = args.repo_base / instance_id
    case_path = args.output_base / instance_id
    output_path = case_path / "output.json"
    if output_path.is_file() and not args.overwrite:
        try:
            completed = json.loads(output_path.read_text(encoding="utf-8")).get("status") == "completed"
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(f"Invalid existing Codex output rerun for {instance_id}: {error}")
        else:
            if completed:
                logger.info(f"Completed instance skipped: {instance_id}")
                return True
    if not repo_path.is_dir():
        logger.error(f"Repository not found: {repo_path}")
        return False
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True)
    atomic_json(case_path / "input_raw.json", instance)
    configuration = {
        "instance_id": instance_id,
        "model": args.model,
        "model_provider": args.model_provider,
        "base_url": args.base_url,
        "sandbox": "read-only",
        "web_search": "disabled",
        "max_workers": args.max_workers,
        "method": "Codex SDK single-turn defect localization",
    }
    atomic_json(case_path / "input.json", configuration)
    started = time.monotonic()
    prompt = f"""You are a fault localization assistant for Python repositories.
Your goal is to identify the exact files, functions, and lines where the bug described in the issue is located. You must not modify any files or run tests.

You have access to standard shell tools and a file viewer. Use them to navigate and read the codebase systematically:
1. Search for relevant identifiers and error messages.
2. Read candidate files and functions.
3. Follow call chains and import paths manually through the source.

A Python repository has been uploaded as the current working directory. The following issue was reported:

<issue>
{instance['problem_statement']}
</issue>

Your task is to identify where the bug is. Do not fix it, modify files, generate a patch, run a repair, run tests, or use network access.

Return at most 10 distinct locations, most suspicious first. Each location must map one repository-relative source file path to one existing physical line number in that file. Use paths relative to the repository root without a leading slash. Verify every path and line against the current repository before returning it. Prefer root-cause decisions and transformations over propagation, exception, definition, documentation, and test lines."""
    atomic_json(case_path / "request.json", {"prompt": prompt, "output_schema": OUTPUT_SCHEMA})
    try:
        os.environ[args.api_key_env] = args.api_key
        if args.model_provider == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key
            overrides = (
                f'openai_base_url="{args.base_url}"',
                'web_search="disabled"',
            )
        else:
            overrides = (
                f'model_providers.{args.model_provider}.name="OpenAI"',
                f'model_providers.{args.model_provider}.base_url="{args.base_url}"',
                f'model_providers.{args.model_provider}.wire_api="responses"',
                f'model_providers.{args.model_provider}.bearer_token_env_var="{args.api_key_env}"',
                f"model_providers.{args.model_provider}.requires_openai_auth=false",
                'web_search="disabled"',
            )
        config = CodexConfig(config_overrides=overrides, cwd=str(repo_path))
        with Codex(config) as codex:
            thread = codex.thread_start(
                model=args.model,
                model_provider=args.model_provider,
                sandbox=Sandbox.read_only,
                ephemeral=True,
            )
            result = thread.run(
                prompt,
                cwd=str(repo_path),
                output_schema=OUTPUT_SCHEMA,
                sandbox=Sandbox.read_only,
            )
        raw_result = {
            "thread_id": thread.id,
            "turn_id": result.id,
            "status": result.status.value,
            "error": result.error.model_dump(mode="json", by_alias=True) if result.error is not None else None,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "duration_ms": result.duration_ms,
            "final_response": result.final_response,
            "items": [item.model_dump(mode="json", by_alias=True) for item in result.items],
            "usage": result.usage.model_dump(mode="json", by_alias=True) if result.usage is not None else None,
        }
        atomic_json(case_path / "raw_output.json", raw_result)
        response = parse_response(result.final_response or "")
        ranked_locations = []
        seen = set()
        for item in response.get("ranked_locations", []):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                logger.warning(f"Invalid Codex location excluded for {instance_id}: {item}")
                continue
            line = item.get("line")
            if not isinstance(line, int) or isinstance(line, bool) or line < 1:
                logger.warning(f"Invalid Codex line excluded for {instance_id}: {item}")
                continue
            path = resolve_location_path(repo_path, item["path"])
            if path is None:
                logger.warning(f"Invalid Codex path excluded for {instance_id}: {item['path']}")
                continue
            key = (path, line)
            source_path = repo_path / path
            if key in seen:
                continue
            try:
                line_count = sum(1 for _ in source_path.open(encoding="utf-8", errors="replace"))
            except OSError as error:
                logger.warning(f"Failed to validate Codex location {instance_id} {path}:{line}: {error}")
                continue
            if line > line_count:
                logger.warning(f"Invalid Codex line excluded for {instance_id}: {path}:{line} > {line_count}")
                continue
            seen.add(key)
            ranked_locations.append({**item, "path": path})
            if len(ranked_locations) == 10:
                break
        usage = result.usage.total.model_dump() if result.usage is not None else None
        completed = result.status == TurnStatus.completed and bool(ranked_locations)
        output = {
            "instance_id": instance_id,
            "status": "completed" if completed else "failed",
            "ranked_locations": ranked_locations,
            "ranked_files": list(dict.fromkeys(item["path"] for item in ranked_locations)),
            "thread_id": thread.id,
            "turn_id": result.id,
            "usage": usage,
            "duration_ms": result.duration_ms,
            "total_seconds": time.monotonic() - started,
        }
        atomic_json(case_path / "output.json", output)
        atomic_json(
            case_path / "trace.json",
            {
                "request": "request.json",
                "raw_output": "raw_output.json",
                "derived_output": "output.json",
            },
        )
        (case_path / "run.log").write_text(
            f"status={output['status']} locations={json.dumps(ranked_locations, ensure_ascii=False)} "
            f"seconds={output['total_seconds']:.2f}\n",
            encoding="utf-8",
        )
        logger.info(f"Completed {instance_id}: {ranked_locations[:2]}")
        return output["status"] == "completed"
    except Exception as error:
        logger.error(f"Codex execution failed for {instance_id}: {error}")
        output = {
            "instance_id": instance_id,
            "status": "failed",
            "ranked_locations": [],
            "ranked_files": [],
            "error": str(error),
            "error_type": type(error).__name__,
            "total_seconds": time.monotonic() - started,
        }
        atomic_json(case_path / "output.json", output)
        atomic_json(
            case_path / "trace.json",
            {
                "request": "request.json",
                "raw_output": "raw_output.json" if (case_path / "raw_output.json").is_file() else None,
                "derived_output": "output.json",
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        (case_path / "run.log").write_text(f"ERROR {type(error).__name__}: {error}\n", encoding="utf-8")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Codex SDK defect localization")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--repo-base", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--base-url", default="http://127.0.0.1:8322/v1")
    parser.add_argument("--api-key", default="sk-1234")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    args.dataset_json = args.dataset_json.expanduser().resolve()
    args.repo_base = args.repo_base.expanduser().resolve()
    args.output_base = args.output_base.expanduser().resolve()
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    instances = selected_instances(dataset, args.instance_ids, args.limit)
    if not instances:
        parser.error("No instances selected")
    logger.info(f"Selected {len(instances)} instances with {min(args.max_workers, len(instances))} workers")
    success_count = 0
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(instances))) as executor:
        futures = {executor.submit(run_instance, instance, args): instance["instance_id"] for instance in instances}
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                success_count += bool(future.result())
            except Exception as error:
                logger.error(f"Worker crashed for {instance_id}: {error}")
            logger.info(f"Progress: {success_count}/{len(instances)} successful")
    return 0 if success_count == len(instances) else 1


if __name__ == "__main__":
    raise SystemExit(main())

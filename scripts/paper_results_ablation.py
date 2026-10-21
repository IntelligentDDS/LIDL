import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger

from paper_results import (
    parse_dataset_patch, functions_for_lines, functions_for_ranked_lines,
    files_for_ranked_lines, limit_ranked_lines, normalize_path as normalize_result_path,
    rank_metrics as unified_rank_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Hydrangea": {
        "dataset_json": PROJECT_ROOT / "datasets" / "Hydrangea" / "hydrangea_defect.json",
        "repo_base": PROJECT_ROOT / "datasets" / "Hydrangea" / "repo",
    },
    "AgentIssue-Bench": {
        "dataset_json": PROJECT_ROOT / "datasets" / "AgentIssue-Bench" / "agent_issue.json",
        "repo_base": PROJECT_ROOT / "datasets" / "AgentIssue-Bench" / "repo",
    },
}
MODELS = [
    "gemma-4-31b",
    "kimi-k2.5",
    "glm-5-2-260617",
    "doubao-seed-2-1-turbo-260628",
    "deepseek-v4-flash-ga-260731",
    "gpt-5.6-terra",
]
VARIANTS = ["full", "without_E", "without_I", "without_R", "without_final", "without_V"]
LEVEL_METRICS = [f"{level} {metric}" for level in ("File", "Function", "Line") for metric in ("Top-1", "Top-3", "Top-5", "MAP", "MRR")]
FILE_CUTOFFS = (1, 3, 5)
LINE_CUTOFFS = (1, 3, 5, 10)
CUTOFFS = FILE_CUTOFFS
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
RAW_FIELDS = [
    "dataset",
    "model",
    "variant",
    "instance_id",
    "status",
    "gold_files",
    "ranked_files",
    "gold_lines",
    "ranked_lines",
    "gold_functions",
    "ranked_functions",
    "evaluation_protocol",
    *LEVEL_METRICS,
    "invalid_prediction_paths",
    "gold_file_count",
    "prediction_count",
    "first_relevant_rank",
    "reciprocal_rank",
    "average_precision",
    *[f"hit_at_{cutoff}" for cutoff in CUTOFFS],
    *[f"line_hit_at_{cutoff}" for cutoff in LINE_CUTOFFS],
    "line_iou",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost",
    "usage_record_count",
    "run_input_tokens",
    "run_output_tokens",
    "run_total_tokens",
    "run_cost",
    "result_path",
    "prediction_source",
    "error",
]
SUMMARY_FIELDS = [
    "scope",
    "scope_value",
    "model",
    "variant",
    "instances",
    "completed_instances",
    "evaluation_protocol",
    *LEVEL_METRICS,
    "failed_instances",
    "missing_instances",
    "labeled_instances",
    "token_instances",
    "cost_instances",
    *[f"hit_at_{cutoff}" for cutoff in CUTOFFS],
    *[f"line_hit_at_{cutoff}" for cutoff in LINE_CUTOFFS],
    "line_iou",
    "map",
    "mrr",
    "mean_input_tokens",
    "mean_output_tokens",
    "mean_total_tokens",
    "sum_input_tokens",
    "sum_output_tokens",
    "sum_total_tokens",
    "mean_cost",
    "sum_cost",
    "mean_run_input_tokens",
    "mean_run_output_tokens",
    "mean_run_total_tokens",
    "mean_run_cost",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_unique(values):
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_path(value):
    path = str(value).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def gold_files_from_patch(patch):
    if isinstance(patch, dict):
        return ordered_unique(normalize_path(path) for path in patch)
    if not isinstance(patch, str):
        logger.warning(f"Unsupported patch type: {type(patch).__name__}")
        return []
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git a/") and " b/" in line:
            files.append(normalize_path(line.split(" b/", 1)[1]))
    return ordered_unique(files)


def gold_lines_from_patch(patch):
    locations = []
    if not isinstance(patch, dict):
        return locations
    for raw_path, value in patch.items():
        path = normalize_path(raw_path)
        changed = []
        context = []
        old_line = None
        lines = value.get("before", []) if isinstance(value, dict) else []
        for line in lines:
            match = HUNK_RE.match(line)
            if match:
                old_line = int(match.group(1))
            elif old_line is not None and line.startswith("-") and not line.startswith("---"):
                changed.append(old_line)
                old_line += 1
            elif old_line is not None and line.startswith("+") and not line.startswith("+++"):
                continue
            elif old_line is not None:
                context.append(old_line)
                old_line += 1
        locations.extend((path, line) for line in changed or context)
    return ordered_unique(locations)


def ranked_paths(values):
    paths = []
    for value in values or []:
        path = value.get("path") if isinstance(value, dict) else value
        if path:
            paths.append(normalize_path(path))
    return ordered_unique(paths)


def ranked_lines(values):
    locations = []
    for value in values or []:
        if not isinstance(value, dict):
            continue
        path = normalize_path(value.get("path", ""))
        line = value.get("line")
        if path and isinstance(line, int) and not isinstance(line, bool):
            locations.append((path, line))
    return ordered_unique(locations)


def load_predictions(case_path, variant):
    output_path = case_path / "output.json"
    validator_path = case_path / "validator_results.json"
    retrieval_path = case_path / "retrieval.json"
    output = read_json(output_path) if output_path.is_file() else {}
    invalid_paths = []
    error = output.get("error")

    if variant == "full":
        values = ranked_lines(output.get("ranked_locations", []))
        values = limit_ranked_lines(values)
        return files_for_ranked_lines(values), values, [], "completed" if output.get("status") == "completed" and values else "failed", "output.ranked_locations", error

    if variant in {"without_E", "without_I", "without_R"}:
        analyzer_path = case_path / "analyzer.json"
        if not analyzer_path.is_file():
            return [], [], [], "missing", "none", "analyzer.json is missing"
        analyzer = read_json(analyzer_path)
        channels = analyzer.get("channels_by_file", {})
        if retrieval_path.is_file():
            retrieval = read_json(retrieval_path)
            full = retrieval.get("variants", {}).get("full", retrieval)
            for item in full.get("file_rankings", []):
                channels[item["path"]] = item.get("retrieval_sources", channels.get(item["path"], []))
        values = [item for item in output.get("ranked_locations", []) if set(channels.get(item.get("path"), [])) & {"E", "I", "R"} != {variant[-1]}]
        predictions = limit_ranked_lines(ranked_lines(values))
        return files_for_ranked_lines(predictions), predictions, [], "completed" if output.get("status") == "completed" else "failed", "output.ranked_locations.posthoc_channel_filter", error

    if variant == "without_final":
        validator = read_json(validator_path).get("full", {}) if validator_path.is_file() else {}
        if "joint_ranked_locations" not in validator:
            return [], [], [], "missing", "none", "joint_ranked_locations is missing"
        values = limit_ranked_lines(ranked_lines(validator["joint_ranked_locations"]))
        return files_for_ranked_lines(values), values, [], "completed" if values else "failed", "validator_results.full.joint_ranked_locations", error

    if variant == "without_V":
        if not retrieval_path.is_file():
            return [], [], [], "missing", "none", error or "retrieval.json is missing"
        retrieval = read_json(retrieval_path)
        variants = retrieval.get("variants", retrieval)
        full = variants.get("full", retrieval)
        line_predictions = ranked_lines(full.get("line_rankings", []))[:10]
        predictions = ordered_unique(path for path, _ in line_predictions)
        status = "completed" if line_predictions else "failed"
        return predictions, line_predictions, [], status, "retrieval.full.line_rankings", error

    if validator_path.is_file():
        validator_results = read_json(validator_path)
        full_result = validator_results.get("full", {})
        values = (
            full_result.get("ranked_locations", [])
            if variant == "full"
            else full_result.get("posthoc_rankings", {}).get(variant, [])
        )
        line_predictions = ranked_lines(values)[:10]
        predictions = ordered_unique(path for path, _ in line_predictions)
        invalid_paths = ordered_unique(
            normalize_path(path) for path in full_result.get("invalid_prediction_paths", [])
        )
        if line_predictions:
            source = "validator_results.full.ranked_locations" if variant == "full" else f"validator_results.full.posthoc_rankings.{variant}"
            return predictions, line_predictions, invalid_paths, "completed", source, error
        if full_result:
            return [], [], invalid_paths, "failed", "validator_results.full", error or "empty ranking"

    rerank_path = case_path / f"rerank_{variant}.json"
    if rerank_path.is_file():
        rerank = read_json(rerank_path)
        line_predictions = ranked_lines(rerank.get("ranked_locations", []))[:10]
        predictions = ordered_unique(path for path, _ in line_predictions)
        status = "completed" if line_predictions else "failed"
        return predictions, line_predictions, [], status, f"rerank_{variant}.ranked_locations", error

    if variant == "full" and output_path.is_file():
        line_predictions = ranked_lines(output.get("ranked_locations", []))[:10]
        predictions = ordered_unique(path for path, _ in line_predictions)
        status = "completed" if output.get("status") == "completed" and line_predictions else "failed"
        return predictions, line_predictions, [], status, "output.ranked_locations", error

    return [], [], [], "missing", "none", error or f"No result found for {variant}"


def variant_from_record(record):
    messages = record.get("data", {}).get("input") or []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except json.JSONDecodeError as error:
            logger.warning(f"Invalid logged user JSON while reading usage: {error}")
            continue
        variant = value.get("variant") if isinstance(value, dict) else None
        if variant in VARIANTS:
            return variant
    return None


def first_numeric(container, paths):
    for path in paths:
        value = container
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def usage_from_response(response):
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = first_numeric(usage, (("prompt_tokens",), ("input_tokens",)))
    output_tokens = first_numeric(usage, (("completion_tokens",), ("output_tokens",)))
    total_tokens = first_numeric(usage, (("total_tokens",),))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    cost = first_numeric(
        response,
        (
            ("response_cost",),
            ("_hidden_params", "response_cost"),
            ("usage", "cost"),
            ("usage", "total_cost"),
        ),
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }


def add_optional(total, key, value):
    if value is None:
        return
    total[key] = (total[key] or 0) + value


def load_usage(case_path):
    by_variant = {
        variant: {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost": None,
            "usage_record_count": 0,
        }
        for variant in VARIANTS
    }
    if (case_path / "calls_llm_joint_validator.jsonl").is_file():
        return by_variant
    new_log_paths = [
        case_path / "calls_llm_file_rerank.jsonl",
        case_path / "calls_llm_file_compare.jsonl",
        case_path / "calls_llm_structure_rerank.jsonl",
        case_path / "calls_llm_structure_global_rerank.jsonl",
        case_path / "calls_llm_line_rerank.jsonl",
        case_path / "calls_llm_line_global_rerank.jsonl",
    ]
    log_paths = (
        [path for path in new_log_paths if path.is_file()]
        or [case_path / "calls_llm_rerank.jsonl"]
    )
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning(f"Invalid usage record {log_path}:{line_number}: {error}")
                continue
            variant = variant_from_record(record)
            response = record.get("data", {}).get("output")
            if variant not in by_variant or not isinstance(response, dict):
                continue
            usage = usage_from_response(response)
            by_variant[variant]["usage_record_count"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens", "cost"):
                add_optional(by_variant[variant], key, usage[key])
    return by_variant


def total_usage(usages):
    result = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "cost": None}
    for variant in EVALUATED_LLM_VARIANTS:
        for key in result:
            add_optional(result, key, usages[variant][key])
    return result


EVALUATED_LLM_VARIANTS = tuple(variant for variant in VARIANTS if variant != "without_V")


def ranking_metrics(gold_files, predictions):
    gold = ordered_unique(gold_files)
    predictions = ordered_unique(predictions)
    if not gold:
        return None
    gold_set = set(gold)
    relevant_seen = 0
    precision_sum = 0.0
    first_rank = None
    for rank, path in enumerate(predictions, 1):
        if path not in gold_set:
            continue
        relevant_seen += 1
        precision_sum += relevant_seen / rank
        if first_rank is None:
            first_rank = rank
    return {
        "first_relevant_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "average_precision": precision_sum / len(gold),
        **{
            f"hit_at_{cutoff}": float(bool(set(predictions[:cutoff]).intersection(gold_set)))
            for cutoff in CUTOFFS
        },
    }


def line_metrics(gold_lines, predictions):
    if not gold_lines:
        return {f"line_hit_at_{cutoff}": None for cutoff in LINE_CUTOFFS} | {"line_iou": None}
    gold_set = set(gold_lines)
    prediction_set = set(predictions)
    union = prediction_set | gold_set
    return {
        **{
            f"line_hit_at_{cutoff}": float(bool(set(predictions[:cutoff]) & gold_set))
            for cutoff in LINE_CUTOFFS
        },
        "line_iou": len(prediction_set & gold_set) / len(union),
    }


def empty_metrics():
    return {
        "first_relevant_rank": None,
        "reciprocal_rank": None,
        "average_precision": None,
        **{f"hit_at_{cutoff}": None for cutoff in CUTOFFS},
    }


def find_case_path(results_root, dataset, model, instance_id):
    return results_root / dataset / model / instance_id


def invalid_prediction_paths(repo_path, predictions):
    return ordered_unique(
        path
        for path in predictions
        if Path(path).is_absolute() or not (repo_path / path).is_file()
    )


def make_rows(results_root, selected_models, limit):
    rows = []
    for dataset, configuration in DATASETS.items():
        instances = read_json(configuration["dataset_json"])
        if limit is not None:
            instances = instances[:limit]
        for model in selected_models:
            for instance in instances:
                instance_id = instance["instance_id"]
                case_path = find_case_path(results_root, dataset, model, instance_id)
                usages = load_usage(case_path)
                run_usage = total_usage(usages)
                gold_files, gold_line_map = parse_dataset_patch(instance)
                gold_lines = [(path, line) for path, lines in gold_line_map.items() for line in lines]
                repo_path = configuration["repo_base"] / instance_id
                gold_functions = functions_for_lines(repo_path, gold_line_map)
                for variant in VARIANTS:
                    predictions, line_predictions, recorded_invalid_paths, status, source, error = load_predictions(case_path, variant)
                    repo_path = configuration["repo_base"] / instance_id
                    line_predictions = limit_ranked_lines([(normalize_result_path(path, instance_id), line) for path, line in line_predictions])
                    predictions = files_for_ranked_lines(line_predictions)
                    predicted_functions = functions_for_ranked_lines(repo_path, line_predictions)
                    level_metrics = {}
                    for level, gold, predicted in (("File", gold_files, predictions), ("Function", gold_functions, predicted_functions), ("Line", gold_lines, line_predictions)):
                        evaluated = unified_rank_metrics(gold, predicted, (1, 3, 5))
                        for cutoff in (1, 3, 5):
                            level_metrics[f"{level} Top-{cutoff}"] = evaluated["hits"][cutoff] if evaluated else 0.0
                        level_metrics[f"{level} MAP"] = evaluated["map"] if evaluated else 0.0
                        level_metrics[f"{level} MRR"] = evaluated["mrr"] if evaluated else 0.0
                    detected_invalid_paths = invalid_prediction_paths(repo_path, predictions)
                    invalid_paths = ordered_unique(recorded_invalid_paths + detected_invalid_paths)
                    if invalid_paths:
                        logger.warning(
                            f"Invalid prediction paths for {dataset} {model} {variant} {instance_id}: {invalid_paths}"
                        )
                    metrics = ranking_metrics(gold_files, predictions) or empty_metrics()
                    metrics.update(line_metrics(gold_lines, line_predictions))
                    usage = usages[variant]
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "variant": variant,
                            "instance_id": instance_id,
                            "status": status,
                            "gold_files": json.dumps(gold_files, ensure_ascii=False),
                            "ranked_files": json.dumps(predictions, ensure_ascii=False),
                            "gold_lines": json.dumps(
                                [{"path": path, "line": line} for path, line in gold_lines],
                                ensure_ascii=False,
                            ),
                            "ranked_lines": json.dumps(
                                [{"path": path, "line": line} for path, line in line_predictions],
                                ensure_ascii=False,
                            ),
                            "invalid_prediction_paths": json.dumps(invalid_paths, ensure_ascii=False),
                            "gold_functions": json.dumps(gold_functions, ensure_ascii=False),
                            "ranked_functions": json.dumps(predicted_functions, ensure_ascii=False),
                            "evaluation_protocol": "saved_run_posthoc_and_stage_comparison",
                            **level_metrics,
                            "gold_file_count": len(gold_files),
                            "prediction_count": len(predictions),
                            **metrics,
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "total_tokens": usage["total_tokens"],
                            "cost": usage["cost"],
                            "usage_record_count": usage["usage_record_count"],
                            "run_input_tokens": run_usage["input_tokens"],
                            "run_output_tokens": run_usage["output_tokens"],
                            "run_total_tokens": run_usage["total_tokens"],
                            "run_cost": run_usage["cost"],
                            "result_path": str(case_path),
                            "prediction_source": source,
                            "error": error,
                        }
                    )
    return rows


def finite_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


def mean_field(rows, field):
    values = finite_values(rows, field)
    return sum(values) / len(values) if values else None


def sum_field(rows, field):
    values = finite_values(rows, field)
    return sum(values) if values else None


def summarize_rows(rows, scope, scope_value):
    summaries = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["variant"])].append(row)
    for (model, variant), selected in sorted(grouped.items()):
        labeled = [row for row in selected if row["gold_file_count"] > 0]
        summary = {
            "scope": scope,
            "scope_value": scope_value,
            "model": model,
            "variant": variant,
            "instances": len(selected),
            "evaluation_protocol": "saved_run_posthoc_and_stage_comparison",
            **{field: mean_field(selected, field) for field in LEVEL_METRICS},
            "completed_instances": sum(row["status"] == "completed" for row in selected),
            "failed_instances": sum(row["status"] == "failed" for row in selected),
            "missing_instances": sum(row["status"] == "missing" for row in selected),
            "labeled_instances": len(labeled),
            "token_instances": len(finite_values(selected, "total_tokens")),
            "cost_instances": len(finite_values(selected, "cost")),
            **{f"hit_at_{cutoff}": mean_field(labeled, f"hit_at_{cutoff}") for cutoff in CUTOFFS},
            **{
                f"line_hit_at_{cutoff}": mean_field(selected, f"line_hit_at_{cutoff}")
                for cutoff in LINE_CUTOFFS
            },
            "line_iou": mean_field(selected, "line_iou"),
            "map": mean_field(labeled, "average_precision"),
            "mrr": mean_field(labeled, "reciprocal_rank"),
            "mean_input_tokens": mean_field(selected, "input_tokens"),
            "mean_output_tokens": mean_field(selected, "output_tokens"),
            "mean_total_tokens": mean_field(selected, "total_tokens"),
            "sum_input_tokens": sum_field(selected, "input_tokens"),
            "sum_output_tokens": sum_field(selected, "output_tokens"),
            "sum_total_tokens": sum_field(selected, "total_tokens"),
            "mean_cost": mean_field(selected, "cost"),
            "sum_cost": sum_field(selected, "cost"),
            "mean_run_input_tokens": mean_field(selected, "run_input_tokens"),
            "mean_run_output_tokens": mean_field(selected, "run_output_tokens"),
            "mean_run_total_tokens": mean_field(selected, "run_total_tokens"),
            "mean_run_cost": mean_field(selected, "run_cost"),
        }
        summaries.append(summary)
    return summaries


def calculate_summaries(rows):
    summaries = summarize_rows(rows, "overall", "all")
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        summaries.extend(summarize_rows(selected, "dataset", dataset))
    return summaries


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate LIDL ablation results")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "results_main",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "paper_results_ablation")
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_models = args.model or MODELS
    rows = make_rows(results_root, selected_models, args.limit)
    summaries = calculate_summaries(rows)
    write_csv(output_dir / "raw.csv", RAW_FIELDS, rows)
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "datasets": list(DATASETS),
                    "models": selected_models,
                    "variants": VARIANTS,
                    "file_cutoffs": FILE_CUTOFFS,
                    "line_cutoffs": LINE_CUTOFFS,
                    "results_root": str(results_root),
                },
                "raw": rows,
                "summary": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Raw rows: {len(rows)}, first two: {rows[:2]}")
    logger.info(f"Summary rows: {len(summaries)}, first two: {summaries[:2]}")
    logger.info(f"Results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

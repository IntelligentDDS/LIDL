import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from loguru import logger

from codekg_retrieval import TREE_SITTER_DEFINITIONS, node_name, tree_sitter_parse, walk_tree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
MODELS = [
    "gemma-4-31b",
    "kimi-k2.5",
    "glm-5-2-260617",
    "doubao-seed-2-1-turbo-260628",
    "deepseek-v4-flash-ga-260731",
    "gpt-5.6-terra",
]
METHODS = [
    "SWE-agent",
    "Agentless",
    "AutoCodeRover",
    "LocAgent",
    "SWE-agent*",
    "Agentless*",
    "Codex",
    "LIDL",
]
RANK_K = {"file": (1, 3, 5), "function": (1, 3, 5), "line": (1, 3, 5)}
MODEL_PRICES_PER_MILLION = {
    "gemma-4-31b": (0.042, 0.22),
    "glm-5-2-260617": (1.40, 4.40),
    "deepseek-v4-flash-ga-260731": (0.44, 1.32),
    "gpt-5.6-terra": (2.00, 12.00),
}
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
LINE_RE = re.compile(r"\bline:\s*(\d+)(?:\s*-\s*(\d+))?", re.IGNORECASE)
LOCAGENT_LINE_RE = re.compile(r"^lines?:")
LOCAGENT_LINE_RANGE_RE = re.compile(r"\s*(\d+)(?:-(\d+))?")
LOCAGENT_PYTHON_PATH_RE = re.compile(r"[\w./-]+\.py")
AUTOCODEROVER_USAGE_RE = re.compile(
    r"API request cost info: input_tokens=(\d+), output_tokens=(\d+), cost=([0-9.eE+-]+)"
)
SWEAGENT_LOCATION_BLOCK = re.compile(
    r"LOCATIONS\s*\n(.*?)\nEND_LOCATIONS",
    re.DOTALL | re.IGNORECASE,
)
SWEAGENT_LOCATION_LINE = re.compile(
    r"(?:file\s*:\s*)?(.+?)\s*,\s*function\s*:\s*(.+?)\s*,\s*line\s*:\s*(\d+)",
    re.IGNORECASE,
)
RAW_FIELDS = [
    "dataset",
    "defect_category",
    "model",
    "approach",
    "instance_id",
    "status",
    "prediction_source",
    "label_files",
    "localized_files",
    "label_functions",
    "localized_functions",
    "label_lines",
    "localized_lines",
    "localized_ranked_lines",
    "input_tokens",
    "output_tokens",
    "token_usage_source",
    "token_usage_scope",
    "cost",
    "result_path",
    "error",
]
STATISTICS_FIELDS = [
    "scope",
    "scope_value",
    "model",
    "approach",
    "instances",
    "completed_instances",
    "token_instances",
    "cost_instances",
    "File Instances",
    "File Top-1",
    "File Top-3",
    "File Top-5",
    "File MAP",
    "File MRR",
    "Function Instances",
    "Function Top-1",
    "Function Top-3",
    "Function Top-5",
    "Function MAP",
    "Function MRR",
    "Line Instances",
    "Line Top-1",
    "Line Top-3",
    "Line Top-5",
    "Line MAP",
    "Line MRR",
    "$Cost",
    "Input Tokens",
    "Output Tokens",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_record(path, instance_id):
    if not path.exists():
        return None
    matched = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("instance_id") == instance_id:
            matched = record
    return matched


def normalize_path(value, instance_id=None, strip_repo_prefix=False):
    path = str(value).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if path.startswith("/repo/"):
        path = path[len("/repo/") :]
    elif strip_repo_prefix and path.startswith("repo/"):
        path = path[len("repo/") :]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if instance_id and path.startswith(f"{instance_id}/"):
        path = path[len(instance_id) + 1 :]
    return path


def ordered_unique(values):
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_hunk_lines(lines):
    changed = []
    context = []
    old_line = None
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
    return ordered_unique(changed or context)


def parse_dataset_patch(instance):
    files = []
    lines = defaultdict(list)
    instance_id = instance["instance_id"]
    for raw_path, patch in instance.get("patch", {}).items():
        path = normalize_path(raw_path, instance_id)
        files.append(path)
        lines[path].extend(parse_hunk_lines(patch.get("before", [])))
    return ordered_unique(files), {path: ordered_unique(values) for path, values in lines.items()}


def collect_functions(source_path):
    if not source_path.exists():
        return []
    try:
        code = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        logger.warning(f"Cannot read source for block mapping {source_path}: {error}")
        return []
    tree, language = tree_sitter_parse(code, source_path)
    if tree is None:
        return [("<module>", 1, max(len(code.splitlines()), 1))]
    source = code.encode("utf-8")
    definitions = TREE_SITTER_DEFINITIONS.get(language, {"classes": set(), "functions": set()})
    blocks = []
    for node in walk_tree(tree.root_node):
        if node.type not in definitions["classes"] | definitions["functions"]:
            continue
        blocks.append(
            (
                node_name(node, source),
                node.start_point.row + 1,
                node.end_point.row + 1,
            )
        )
    blocks.append(("<module>", 1, max(len(code.splitlines()), 1)))
    return blocks


def functions_for_lines(repo_path, line_locations):
    mapped = []
    for relative_path, line_numbers in line_locations.items():
        functions = collect_functions(repo_path / relative_path)
        for line_number in line_numbers:
            candidates = [item for item in functions if item[1] <= line_number <= item[2]]
            selected = min(candidates, key=lambda item: item[2] - item[1], default=None)
            if selected:
                mapped.append((relative_path, selected[0]))
    return ordered_unique(mapped)


def functions_for_ranked_lines(repo_path, ranked_lines):
    mapped = []
    cache = {}
    for relative_path, line_number in ranked_lines:
        functions = cache.setdefault(
            relative_path, collect_functions(repo_path / relative_path)
        )
        candidates = [item for item in functions if item[1] <= line_number <= item[2]]
        selected = min(candidates, key=lambda item: item[2] - item[1], default=None)
        if selected:
            mapped.append((relative_path, selected[0]))
    return ordered_unique(mapped)


def files_for_ranked_lines(ranked_lines):
    return ordered_unique(path for path, _ in ranked_lines)


def parse_agentless_lines(raw_locations, instance_id):
    lines = defaultdict(list)
    for raw_path, entries in (raw_locations or {}).items():
        path = normalize_path(raw_path, instance_id)
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries or []:
            for start, end in LINE_RE.findall(entry):
                lines[path].extend(range(int(start), int(end or start) + 1))
    return {path: ordered_unique(values) for path, values in lines.items()}


def flatten_lines(line_locations):
    return [(path, line) for path, values in line_locations.items() for line in values]


def limit_ranked_lines(ranked_lines, limit=10):
    limited = []
    seen = set()
    for path, line_number in ranked_lines:
        location = (path, line_number)
        if location in seen:
            continue
        seen.add(location)
        limited.append(location)
        if len(limited) == limit:
            break
    return limited


def group_lines(ranked_lines):
    grouped = defaultdict(list)
    for path, line_number in ranked_lines:
        grouped[path].append(line_number)
    return dict(grouped)


def sum_usage(items):
    totals = defaultdict(float)

    def visit(value):
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict):
                for key, number in usage.items():
                    if isinstance(number, (int, float)):
                        totals[key] += number
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(items)
    return dict(totals)


def relative_path(path):
    return str(path.relative_to(PROJECT_ROOT)) if path and path.exists() else None


def empty_result(status, error=None):
    return {
        "status": status,
        "prediction_source": "none",
        "localized_files": [],
        "localized_functions": [],
        "localized_lines": {},
        "localized_ranked_lines": [],
        "input_tokens": None,
        "output_tokens": None,
        "token_usage_source": None,
        "token_usage_scope": None,
        "cost": None,
        "result_path": None,
        "error": error,
    }


def read_agentless_result(result_root, dataset, model, instance_id):
    run_dir = PROJECT_ROOT / "results" / result_root / dataset / model / instance_id
    file_level = read_jsonl_record(run_dir / "file_level/loc_outputs.jsonl", instance_id)
    irrelevant = read_jsonl_record(run_dir / "file_level_irrelevant/loc_outputs.jsonl", instance_id)
    retrieval = read_jsonl_record(run_dir / "retrievel_embedding/retrieve_locs.jsonl", instance_id)
    related = read_jsonl_record(run_dir / "related_elements/loc_outputs.jsonl", instance_id)
    edit = read_jsonl_record(run_dir / "edit_location_individual/loc_merged_0-0_outputs.jsonl", instance_id)
    usage = sum_usage(
        [
            (file_level or {}).get("file_traj"),
            (irrelevant or {}).get("file_traj"),
            (retrieval or {}).get("traj"),
            (related or {}).get("related_loc_traj"),
            (edit or {}).get("edit_loc_traj"),
        ]
    )
    ranked_lines = limit_ranked_lines(
        flatten_lines(parse_agentless_lines((edit or {}).get("found_edit_locs"), instance_id))
    )
    localized_lines = group_lines(ranked_lines)
    completed = bool(ranked_lines)
    return {
        "status": "completed" if completed else "missing_or_incomplete",
        "prediction_source": "localization" if completed else "none",
        "localized_files": files_for_ranked_lines(ranked_lines),
        "localized_functions": [],
        "localized_lines": localized_lines,
        "localized_ranked_lines": ranked_lines,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "token_usage_source": "agentless_trajectory_usage",
        "token_usage_scope": "framework_reported_localization_api_calls_excluding_embeddings",
        "cost": None,
        "result_path": relative_path(run_dir),
    }


def find_autocoderover_run(dataset, model, instance_id):
    root = PROJECT_ROOT / "results/results_autocoderover" / dataset / model
    runs = sorted(root.glob(f"{instance_id}_*")) if root.exists() else []
    return runs[-1] if runs else None


def read_autocoderover_usage(run_dir):
    cost_path = run_dir / "cost.json"
    if cost_path.exists():
        usage = read_json(cost_path)
        return usage.get("total_input_tokens"), usage.get("total_output_tokens"), usage.get("total_cost")
    info_path = run_dir / "info.log"
    if not info_path.exists():
        return None, None, None
    matches = AUTOCODEROVER_USAGE_RE.findall(info_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return None, None, None
    return (
        sum(int(input_tokens) for input_tokens, _, _ in matches),
        sum(int(output_tokens) for _, output_tokens, _ in matches),
        sum(float(cost) for _, _, cost in matches),
    )


def read_autocoderover_locations(output_dir):
    if output_dir is None:
        return []
    for name in ("bug_locations_after_process.json", "bug_locations_before_process.json"):
        path = output_dir / "search" / name
        if path.exists():
            return read_json(path)
    return []


def parse_autocoderover_lines(locations, instance_id):
    ranked_lines = []
    for location in locations:
        path = normalize_path(location.get("rel_file_path", ""), instance_id)
        if not path:
            continue
        line_number = location.get("line")
        if not isinstance(line_number, int) or isinstance(line_number, bool):
            line_number = location.get("start")
        if isinstance(line_number, int) and not isinstance(line_number, bool):
            ranked_lines.append((path, line_number))
    return ranked_lines


def read_autocoderover(dataset, model, instance_id):
    run_dir = find_autocoderover_run(dataset, model, instance_id)
    if run_dir is None:
        return empty_result("missing_or_incomplete")
    locations = []
    for output_dir in reversed(sorted(path for path in run_dir.glob("output_*") if path.is_dir())):
        locations = read_autocoderover_locations(output_dir)
        if locations:
            break
    ranked_lines = limit_ranked_lines(parse_autocoderover_lines(locations, instance_id))
    lines = group_lines(ranked_lines)
    input_tokens, output_tokens, cost = read_autocoderover_usage(run_dir)
    completed = bool(ranked_lines)
    return {
        "status": "completed" if completed else "failed_or_missing",
        "prediction_source": "localization" if completed else "none",
        "localized_files": files_for_ranked_lines(ranked_lines),
        "localized_functions": [],
        "localized_lines": lines,
        "localized_ranked_lines": ranked_lines,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_usage_source": "autocoderover_cost_json_or_info_log",
        "token_usage_scope": "framework_reported_cumulative_api_usage",
        "cost": cost,
        "result_path": relative_path(run_dir),
    }


def find_sweagent_run(result_root, dataset, model, instance_id):
    model_roots = [model, f"openai-{model}"]
    for model_root in model_roots:
        root = PROJECT_ROOT / "results" / result_root / dataset / model_root / instance_id
        if not root.exists():
            continue
        runs = [path for path in root.iterdir() if path.is_dir()]
        if runs:
            return max(runs, key=lambda path: path.stat().st_mtime)
    return None


def parse_sweagent_lines(trajectory, instance_id):
    texts = []
    for item in reversed((trajectory or {}).get("history", [])):
        content = item.get("content")
        if isinstance(content, str) and "LOCATIONS" in content.upper():
            texts.append(content)
    for step in reversed((trajectory or {}).get("trajectory", [])):
        for field in ("observation", "response", "action"):
            content = step.get(field)
            if isinstance(content, str) and "LOCATIONS" in content.upper():
                texts.append(content)
    for text in texts:
        match = SWEAGENT_LOCATION_BLOCK.search(text)
        if not match:
            continue
        ranked_lines = []
        for raw_line in match.group(1).splitlines():
            location = SWEAGENT_LOCATION_LINE.search(raw_line)
            if not location:
                continue
            path = normalize_path(
                location.group(1), instance_id, strip_repo_prefix=True
            )
            ranked_lines.append((path, int(location.group(3))))
        if ranked_lines:
            return ranked_lines
    return []


def read_sweagent_result(result_root, dataset, model, instance_id):
    run_dir = find_sweagent_run(result_root, dataset, model, instance_id)
    if run_dir is None:
        return empty_result("missing_or_incomplete")
    trajectories = sorted(run_dir.glob("*.traj"))
    trajectory = read_json(trajectories[-1]) if trajectories else None
    ranked_lines = limit_ranked_lines(parse_sweagent_lines(trajectory, instance_id))
    lines = group_lines(ranked_lines)
    usage = (trajectory or {}).get("info", {}).get("model_stats", {})
    completed = bool(ranked_lines)
    return {
        "status": "completed" if completed else "failed_or_missing",
        "prediction_source": "localization" if completed else "none",
        "localized_files": files_for_ranked_lines(ranked_lines),
        "localized_functions": [],
        "localized_lines": lines,
        "localized_ranked_lines": ranked_lines,
        "input_tokens": usage.get("tokens_sent"),
        "output_tokens": usage.get("tokens_received"),
        "token_usage_source": "sweagent_traj_info_model_stats",
        "token_usage_scope": "framework_reported_cumulative_api_usage",
        "cost": usage.get("instance_cost"),
        "result_path": relative_path(run_dir),
    }


@lru_cache(maxsize=None)
def locagent_valid_files(repo_path):
    valid_files = {
        str(path.relative_to(repo_path)).replace("\\", "/")
        for path in repo_path.rglob("*.py")
        if path.is_file()
    }
    return valid_files, ordered_unique(path.split("/", 1)[0] for path in valid_files)


def locagent_sample_lines(raw_output, repo_path, instance_id):
    valid_files, valid_top_folders = locagent_valid_files(repo_path)
    current_file = None
    locations = []
    for raw_line in raw_output.strip("` \n").splitlines():
        line = raw_line.strip().strip(":").strip()
        if not line:
            continue
        if line.endswith(".py"):
            match = LOCAGENT_PYTHON_PATH_RE.search(line)
            current_file = None
            if not match:
                continue
            matched_path = normalize_path(match.group(0), instance_id)
            if matched_path in valid_files:
                current_file = matched_path
                continue
            starts = [
                matched_path.index(f"{folder}/")
                for folder in valid_top_folders
                if f"{folder}/" in matched_path
            ]
            if not starts:
                continue
            path = matched_path[min(starts) :]
            if path in valid_files:
                current_file = path
            continue
        if current_file is None or not LOCAGENT_LINE_RE.match(line):
            continue
        for start, end in LOCAGENT_LINE_RANGE_RE.findall(line.split(":", 1)[1]):
            for line_number in range(int(start), int(end or start) + 1):
                location = (current_file, line_number)
                if location not in locations:
                    locations.append(location)
    return locations


def merge_locagent_raw_lines(raw_outputs, repo_path, instance_id):
    weights = defaultdict(float)
    for raw_output in raw_outputs or []:
        if not isinstance(raw_output, str):
            continue
        for rank, location in enumerate(
            locagent_sample_lines(raw_output, repo_path, instance_id), start=1
        ):
            weights[location] += 1 / rank
    return [
        location
        for location, _ in sorted(
            weights.items(), key=lambda item: item[1], reverse=True
        )[:10]
    ]


def read_locagent(dataset, model, instance_id):
    run_dir = PROJECT_ROOT / "results/results_locagent" / dataset / model
    raw = read_jsonl_record(run_dir / "loc_outputs.jsonl", instance_id)
    trajectory = read_jsonl_record(run_dir / "loc_trajs.jsonl", instance_id)
    dataset_config = next(item for item in DATASETS if item["name"] == dataset)
    repo_path = dataset_config["repo_base"] / instance_id
    ranked_lines = (
        []
        if raw is None or raw.get("found_files") == [[]]
        else merge_locagent_raw_lines(
            raw.get("raw_output_loc", []), repo_path, instance_id
        )
    )
    localized_lines = group_lines(ranked_lines)
    usage = (trajectory or {}).get("usage", {})
    completed = raw is not None and bool(ranked_lines)
    return {
        "status": "completed" if completed else "failed_or_missing",
        "prediction_source": "loc_outputs.raw_output_loc_mrr" if completed else "none",
        "localized_files": files_for_ranked_lines(ranked_lines),
        "localized_functions": [],
        "localized_lines": localized_lines,
        "localized_ranked_lines": ranked_lines,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "token_usage_source": "locagent_loc_trajs_top_level_usage",
        "token_usage_scope": "framework_reported_cumulative_multiturn_api_usage",
        "cost": usage.get("cost($)"),
        "result_path": relative_path(run_dir),
    }


def read_codex(dataset, model, instance_id):
    run_dir = PROJECT_ROOT / "results" / "results_codex" / dataset / model / instance_id
    output_path = run_dir / "output.json"
    if not output_path.is_file():
        return empty_result("missing_or_incomplete")
    output = read_json(output_path)
    ranked_lines = limit_ranked_lines(
        [
            (normalize_path(item.get("path", ""), instance_id), item.get("line"))
            for item in output.get("ranked_locations", [])
            if isinstance(item, dict)
            and item.get("path")
            and isinstance(item.get("line"), int)
            and not isinstance(item.get("line"), bool)
        ]
    )
    usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
    completed = output.get("status") == "completed" and bool(ranked_lines)
    return {
        "status": "completed" if completed else "failed_or_missing",
        "prediction_source": "output.ranked_locations" if ranked_lines else "none",
        "localized_files": files_for_ranked_lines(ranked_lines),
        "localized_functions": [],
        "localized_lines": group_lines(ranked_lines),
        "localized_ranked_lines": ranked_lines,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "token_usage_source": "codex_sdk_thread_usage",
        "token_usage_scope": "single_codex_localization_thread",
        "cost": None,
        "result_path": relative_path(run_dir),
        "error": output.get("error"),
    }


def read_lidl(dataset, model, instance_id):
    run_dir = PROJECT_ROOT / "results" / "results_main" / dataset / model / instance_id
    output_path = run_dir / "output.json"
    if not output_path.is_file():
        return empty_result("missing_or_incomplete")
    output = read_json(output_path)
    ranked_lines = limit_ranked_lines(
        [
            (normalize_path(item.get("path", ""), instance_id), item.get("line"))
            for item in output.get("ranked_locations", [])
            if isinstance(item, dict)
            and item.get("path")
            and isinstance(item.get("line"), int)
            and not isinstance(item.get("line"), bool)
        ]
    )
    localized_files = files_for_ranked_lines(ranked_lines)
    input_tokens = None
    output_tokens = None
    cost = None
    current_log_paths = [
        run_dir / "calls_llm_annotation.jsonl",
        run_dir / "calls_llm_analyzer.jsonl",
        run_dir / "calls_llm_joint_validator.jsonl",
    ]
    legacy_log_paths = [
        run_dir / "calls_llm_file_rerank.jsonl",
        run_dir / "calls_llm_file_compare.jsonl",
        run_dir / "calls_llm_structure_rerank.jsonl",
        run_dir / "calls_llm_structure_global_rerank.jsonl",
        run_dir / "calls_llm_line_rerank.jsonl",
        run_dir / "calls_llm_line_global_rerank.jsonl",
    ]
    log_paths = [path for path in current_log_paths if path.is_file()]
    if not log_paths:
        log_paths = (
            [path for path in legacy_log_paths if path.is_file()]
            or [run_dir / "calls_llm_rerank.jsonl"]
        )
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line).get("data", {})
            except json.JSONDecodeError as error:
                logger.warning(f"Invalid LIDL API record {log_path}:{line_number}: {error}")
                continue
            response = record.get("output")
            if not isinstance(response, dict):
                continue
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            record_input = usage.get("prompt_tokens", usage.get("input_tokens"))
            record_output = usage.get("completion_tokens", usage.get("output_tokens"))
            if isinstance(record_input, (int, float)) and not isinstance(record_input, bool):
                input_tokens = (input_tokens or 0) + record_input
            if isinstance(record_output, (int, float)) and not isinstance(record_output, bool):
                output_tokens = (output_tokens or 0) + record_output
            record_cost = response.get("response_cost")
            if record_cost is None and isinstance(response.get("_hidden_params"), dict):
                record_cost = response["_hidden_params"].get("response_cost")
            if isinstance(record_cost, (int, float)) and not isinstance(record_cost, bool):
                cost = (cost or 0) + record_cost
    completed = output.get("status") == "completed" and bool(ranked_lines)
    return {
        "status": "completed" if completed else "failed_or_missing",
        "prediction_source": "output.ranked_locations" if ranked_lines else "none",
        "localized_files": localized_files,
        "localized_functions": [],
        "localized_lines": group_lines(ranked_lines),
        "localized_ranked_lines": ranked_lines,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_usage_source": "lidl_api_usage",
        "token_usage_scope": "annotation_analyzer_and_joint_validator_api_calls",
        "cost": cost,
        "result_path": relative_path(run_dir),
        "error": output.get("error"),
    }


def json_files(values):
    return json.dumps(values, ensure_ascii=False)


def json_functions(values):
    return json.dumps([{"path": path, "name": name} for path, name in values], ensure_ascii=False)


def json_lines(values):
    return json.dumps(values, ensure_ascii=False)


def json_ranked_lines(values):
    items = [{"path": path, "line": line_number} for path, line_number in values]
    return json.dumps(items, ensure_ascii=False)


def invalid_prediction_paths(repo_path, ranked_lines):
    return ordered_unique(
        path
        for path, _ in ranked_lines
        if Path(path).is_absolute() or not (repo_path / path).is_file()
    )


def extract_case(case):
    instance = case["instance"]
    readers = {
        "Agentless": lambda dataset, model, instance_id: read_agentless_result(
            "results_agentless", dataset, model, instance_id
        ),
        "Agentless*": lambda dataset, model, instance_id: read_agentless_result(
            "results_agentless_repograph", dataset, model, instance_id
        ),
        "AutoCodeRover": read_autocoderover,
        "LocAgent": read_locagent,
        "SWE-agent": lambda dataset, model, instance_id: read_sweagent_result(
            "results_sweagent", dataset, model, instance_id
        ),
        "SWE-agent*": lambda dataset, model, instance_id: read_sweagent_result(
            "results_sweagent_repograph", dataset, model, instance_id
        ),
        "Codex": read_codex,
        "LIDL": read_lidl,
    }
    result = readers[case["approach"]](case["dataset"], case["model"], instance["instance_id"])
    repo_path = case["repo_base"] / instance["instance_id"]
    label_files, label_lines = parse_dataset_patch(instance)
    label_functions = functions_for_lines(repo_path, label_lines)
    ranked_lines = result["localized_ranked_lines"]
    localized_files = files_for_ranked_lines(ranked_lines)
    localized_lines = group_lines(ranked_lines)
    localized_functions = functions_for_ranked_lines(repo_path, ranked_lines)
    invalid_paths = invalid_prediction_paths(repo_path, ranked_lines)
    if invalid_paths:
        logger.warning(
            f"Invalid prediction paths for {case['approach']} {instance['instance_id']}: "
            f"{invalid_paths}"
        )
    status = result["status"]
    prediction_source = result["prediction_source"]
    return {
        "dataset": case["dataset"],
        "defect_category": case["defect_category"],
        "model": case["model"],
        "approach": case["approach"],
        "instance_id": instance["instance_id"],
        "status": status,
        "prediction_source": prediction_source,
        "label_files": json_files(label_files),
        "localized_files": json_files(localized_files),
        "label_functions": json_functions(label_functions),
        "localized_functions": json_functions(localized_functions),
        "label_lines": json_lines(label_lines),
        "localized_lines": json_lines(localized_lines),
        "localized_ranked_lines": json_ranked_lines(ranked_lines),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "token_usage_source": result["token_usage_source"],
        "token_usage_scope": result["token_usage_scope"],
        "cost": result["cost"],
        "result_path": result["result_path"],
        "error": (
            f"invalid_prediction_paths: {json.dumps(invalid_paths, ensure_ascii=False)}"
            if invalid_paths
            else result.get("error")
        ),
    }


def extraction_error(case, error):
    return {
        "dataset": case["dataset"],
        "defect_category": case["defect_category"],
        "model": case["model"],
        "approach": case["approach"],
        "instance_id": case["instance"]["instance_id"],
        "status": "extraction_error",
        "prediction_source": "none",
        "label_files": "[]",
        "localized_files": "[]",
        "label_functions": "[]",
        "localized_functions": "[]",
        "label_lines": "{}",
        "localized_lines": "{}",
        "localized_ranked_lines": "[]",
        "input_tokens": None,
        "output_tokens": None,
        "token_usage_source": None,
        "token_usage_scope": None,
        "cost": None,
        "result_path": None,
        "error": f"{type(error).__name__}: {error}",
    }


def parse_files(value):
    return ordered_unique(json.loads(value or "[]"))


def parse_functions(value):
    return ordered_unique((item["path"], item["name"]) for item in json.loads(value or "[]"))


def parse_lines(value):
    locations = json.loads(value or "{}")
    return ordered_unique((path, line) for path, lines in locations.items() for line in lines)


def parse_ranked_lines(value):
    return ordered_unique(
        (item["path"], item["line"]) for item in json.loads(value or "[]")
    )


def rank_metrics(labels, predictions, ks):
    labels = ordered_unique(labels)
    predictions = ordered_unique(predictions)
    if not labels:
        return None
    label_set = set(labels)
    first_rank = next((rank for rank, value in enumerate(predictions, 1) if value in label_set), None)
    relevant_seen = 0
    precision_sum = 0.0
    for rank, value in enumerate(predictions, 1):
        if value in label_set:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    return {
        "hits": {k: float(bool(set(predictions[:k]) & label_set)) for k in ks},
        "map": precision_sum / len(label_set),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "iou": len(set(predictions) & label_set) / len(set(predictions) | label_set),
    }


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def numeric_mean(rows, field):
    values = [float(row[field]) for row in rows if row[field] not in (None, "")]
    return mean(values)


def token_cost(model, input_tokens, output_tokens):
    rates = MODEL_PRICES_PER_MILLION.get(model)
    if rates is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


def summarize_group(scope, scope_value, rows):
    summaries = []
    parsers = {"File": parse_files, "Function": parse_functions, "Line": parse_lines}
    prediction_parsers = {
        "File": parse_files,
        "Function": parse_functions,
        "Line": parse_ranked_lines,
    }
    raw_fields = {
        "File": ("label_files", "localized_files"),
        "Function": ("label_functions", "localized_functions"),
        "Line": ("label_lines", "localized_ranked_lines"),
    }
    for model in MODELS:
        for approach in METHODS:
            selected = [row for row in rows if row["model"] == model and row["approach"] == approach]
            if not selected:
                continue
            completed = [row for row in selected if row["status"] == "completed"]
            summary = {
                "scope": scope,
                "scope_value": scope_value,
                "model": model,
                "approach": approach,
                "instances": len(selected),
                "completed_instances": len(completed),
                "token_instances": sum(
                    row["input_tokens"] not in (None, "") and row["output_tokens"] not in (None, "")
                    for row in selected
                ),
                "cost_instances": sum(row["cost"] not in (None, "") for row in selected),
            }
            for label, parser in parsers.items():
                label_field, prediction_field = raw_fields[label]
                metric_rows = selected
                metrics = []
                for row in metric_rows:
                    labels = parser(row[label_field])
                    if labels:
                        predictions = (
                            prediction_parsers[label](row[prediction_field])
                            if row["status"] == "completed"
                            else []
                        )
                        metrics.append(
                            rank_metrics(labels, predictions, RANK_K[label.lower()])
                        )
                summary[f"{label} Instances"] = len(metrics)
                for k in RANK_K[label.lower()]:
                    summary[f"{label} Top-{k}"] = mean([metric["hits"][k] for metric in metrics])
                summary[f"{label} MAP"] = mean([metric["map"] for metric in metrics])
                summary[f"{label} MRR"] = mean([metric["mrr"] for metric in metrics])
            summary["Input Tokens"] = numeric_mean(selected, "input_tokens")
            summary["Output Tokens"] = numeric_mean(selected, "output_tokens")
            summary["$Cost"] = token_cost(
                model,
                summary["Input Tokens"],
                summary["Output Tokens"],
            )
            summaries.append(summary)
    return summaries


def calculate_statistics(rows):
    statistics = summarize_group("overall", "all", rows)
    for dataset in [item["name"] for item in DATASETS]:
        statistics.extend(summarize_group("dataset", dataset, [row for row in rows if row["dataset"] == dataset]))
    categories = sorted({row["defect_category"] for row in rows if row["defect_category"]})
    for category in categories:
        statistics.extend(
            summarize_group("category", category, [row for row in rows if row["defect_category"] == category])
        )
    return statistics


def load_categories():
    path = PROJECT_ROOT / "scripts/classification_agreed.csv"
    with path.open(encoding="utf-8", newline="") as file:
        return {
            (row["benchmark"], row["instance_id"]): row["defect_category"]
            for row in csv.DictReader(file)
        }


def load_raw(path):
    if not path.exists():
        return {}
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as file:
        return {
            (row["dataset"], row["model"], row["approach"], row["instance_id"]): row
            for row in csv.DictReader(file)
        }


def write_csv(path, fields, rows, decimal_places=None):
    if decimal_places is not None:
        rows = [
            {
                key: f"{value:.{decimal_places}f}"
                if isinstance(value, float)
                else value
                for key, value in row.items()
            }
            for row in rows
        ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract paper results or calculate statistics")
    parser.add_argument("stage", choices=("extract", "statistics"))
    parser.add_argument("--limit", type=int, default=0, help="The first N instances of each data set, 0 means all")
    parser.add_argument("--max-workers", type=int, default=min(50, (os.cpu_count() or 1) * 4))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/paper_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", choices=MODELS, help="Extract only this model, preserving other existing rows")
    parser.add_argument("--approach", choices=METHODS, help="Extract only this method, preserving other existing rows")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.csv"
    statistics_path = output_dir / "statistics.csv"

    if args.stage == "statistics":
        if not raw_path.is_file():
            parser.error(f"Raw results do not exist: {raw_path}")
        rows = list(load_raw(raw_path).values())
        rows.sort(key=lambda row: (row["dataset"], row["model"], row["approach"], row["instance_id"]))
        statistics = calculate_statistics(rows)
        write_csv(statistics_path, STATISTICS_FIELDS, statistics, decimal_places=4)
        logger.info(f"Statistics rows: {len(statistics)}, first two: {statistics[:2]}")
        logger.info(f"Statistics written to {statistics_path}")
        return 0

    targeted = args.model is not None or args.approach is not None
    existing = load_raw(raw_path) if args.resume or targeted else {}
    categories = load_categories()
    cases = []
    for dataset in DATASETS:
        instances = read_json(dataset["dataset_json"])
        if args.limit:
            instances = instances[: args.limit]
        for instance in instances:
            for model in ([args.model] if args.model else MODELS):
                for approach in ([args.approach] if args.approach else METHODS):
                    key = (dataset["name"], model, approach, instance["instance_id"])
                    if key not in existing or (targeted and not args.resume):
                        cases.append(
                            {
                                "dataset": dataset["name"],
                                "repo_base": dataset["repo_base"],
                                "model": model,
                                "approach": approach,
                                "instance": instance,
                                "defect_category": categories.get((dataset["name"], instance["instance_id"])),
                            }
                        )
    logger.info(f"Cases: total={len(existing) + len(cases)}, completed={len(existing)}, pending={len(cases)}")

    rows = list(existing.values())
    mode = "a" if (args.resume or targeted) and raw_path.exists() else "w"
    with raw_path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        if mode == "w":
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(extract_case, case): case for case in cases}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    logger.error(
                        f"Extraction failed for {case['dataset']} {case['model']} "
                        f"{case['approach']} {case['instance']['instance_id']}: {error}"
                    )
                    row = extraction_error(case, error)
                writer.writerow(row)
                file.flush()
                rows.append(row)
                if index % 100 == 0 or index == len(cases):
                    logger.info(f"Progress: {index}/{len(cases)} cases extracted")

    logger.info(f"Extracted {len(cases)} cases; first two: {rows[:2]}")
    rows = list(load_raw(raw_path).values())
    rows.sort(key=lambda row: (row["dataset"], row["model"], row["approach"], row["instance_id"]))
    compact_path = raw_path.with_suffix(".csv.tmp")
    write_csv(compact_path, RAW_FIELDS, rows)
    compact_path.replace(raw_path)
    logger.info(f"Raw rows: {len(rows)} -> {raw_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import ast
import csv
import math
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "gemma-4-31b",
    "kimi-k2.5",
    "glm-5-2-260617",
    "doubao-seed-2-1-turbo-260628",
    "deepseek-v4-flash-ga-260731",
    "gpt-5.6-terra",
]
MODEL_DISPLAY_NAMES = {
    "gemma-4-31b": "Gemma-4-31B",
    "kimi-k2.5": "Kimi-K2.5",
    "glm-5-2-260617": "GLM-5-2",
    "doubao-seed-2-1-turbo-260628": "Doubao-Seed-2.1",
    "deepseek-v4-flash-ga-260731": "DeepSeek-V4-Flash",
    "gpt-5.6-terra": "GPT-5.6-Terra",
}
MODEL_LATEX_NAMES = {
    **MODEL_DISPLAY_NAMES,
    "deepseek-v4-flash-ga-260731": r"\shortstack[l]{DeepSeek-\\V4-Flash}",
}
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
PERFORMANCE_MODELS = [
    "gemma-4-31b",
    "glm-5-2-260617",
    "deepseek-v4-flash-ga-260731",
    "gpt-5.6-terra",
]
PERFORMANCE_METHODS = [
    "SWE-agent",
    "Agentless",
    "AutoCodeRover",
    "SWE-agent*",
    "Agentless*",
    "LocAgent",
    "Codex",
    "LIDL",
]
METHOD_DISPLAY_NAMES = {
    "SWE-agent": "SWE-agent",
    "Agentless": "Agentless",
    "AutoCodeRover": "Auto-Code-Rover",
    "SWE-agent*": "SWE-agent*",
    "Agentless*": "Agentless*",
    "LocAgent": "LocAgent",
    "Codex": "Codex",
    "LIDL": "LIDL",
}
METHOD_LATEX = {
    "SWE-agent": "SWE-agent",
    "Agentless": "Agentless",
    "AutoCodeRover": "Auto-Code-Rover",
    "LocAgent": "LocAgent",
    "SWE-agent*": "SWE-agent$^*$",
    "Agentless*": "Agentless$^*$",
    "Codex": "Codex",
    "LIDL": "LIDL",
}
APPROACH_CHARACTERISTICS = [
    ("SWE-agent", True, False, False, False),
    ("Agentless", True, False, False, True),
    ("AutoCodeRover", True, False, False, True),
    ("SWE-agent*", True, True, False, False),
    ("Agentless*", True, True, False, True),
    ("LIDL", True, True, True, True),
]
PERFORMANCE_COLUMNS = [
    "model",
    "approach",
    "File Top-1",
    "File Top-3",
    "File Top-5",
    "File MAP",
    "File MRR",
    "Function Top-1",
    "Function Top-3",
    "Function Top-5",
    "Function MAP",
    "Function MRR",
    "Line Top-1",
    "Line Top-3",
    "Line Top-5",
    "Line MAP",
    "Line MRR",
    "Input Tokens",
    "Output Tokens",
    "$Cost",
]
PERFORMANCE_HIGHER_IS_BETTER = PERFORMANCE_COLUMNS[2:17]
ABLATION_VARIANTS = ["without_E", "without_I", "without_R", "without_final", "without_V", "full"]
ABLATION_NAMES = {
    "without_E": r"LIDL: filter $\mathcal{E}$-only",
    "without_I": r"LIDL: filter $\mathcal{I}$-only",
    "without_R": r"LIDL: filter $\mathcal{R}$-only",
    "without_final": "Joint Validator only",
    "without_V": "Retrieval only",
    "full": "LIDL",
}


def write_rows(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def bold_if_best(value, best, formatted):
    if is_finite(value) and is_finite(best) and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12):
        return rf"\textbf{{{formatted}}}"
    return formatted


def format_score(value):
    return "--" if not is_finite(value) else f"{value:.4f}".rstrip("0").rstrip(".")


def format_cost(value):
    return "--" if not is_finite(value) else f"{value:.3f}"


def format_tokens(value):
    return "--" if not is_finite(value) else f"{value / 1000:.2f}"


def build_approach_table(output_dir):
    fields = ["Approach", "Repo Structure", "Repo Graph", "Code Semantic Analysis", "Multi Stage"]
    rows = [
        {
            "Approach": approach,
            "Repo Structure": repo_structure,
            "Repo Graph": repo_graph,
            "Code Semantic Analysis": semantic,
            "Multi Stage": multi_stage,
        }
        for approach, repo_structure, repo_graph, semantic, multi_stage in APPROACH_CHARACTERISTICS
    ]
    write_rows(output_dir / "approach_comparison.csv", fields, rows)
    mark = lambda enabled: r"\textcolor{blue}{\ding{52}}" if enabled else r"\textcolor{red}{\ding{56}}"
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Approach & Repo Structure & Repo Graph & Code Semantic Analysis & Multi Stage \\\\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        if row["Approach"] == "LIDL":
            lines.append(r"\midrule")
        name = METHOD_LATEX[row["Approach"]]
        values = [mark(row[field]) for field in fields[1:]]
        lines.append(f"{name} & " + " & ".join(values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "approach_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def select_performance_rows(statistics_path):
    statistics = pd.read_csv(statistics_path)
    selected = statistics[
        (statistics["scope"] == "overall")
        & (statistics["scope_value"] == "all")
        & (statistics["model"].isin(PERFORMANCE_MODELS))
        & (statistics["approach"].isin(PERFORMANCE_METHODS))
    ].copy()
    selected["model"] = pd.Categorical(
        selected["model"], categories=PERFORMANCE_MODELS, ordered=True
    )
    selected["approach"] = pd.Categorical(
        selected["approach"], categories=PERFORMANCE_METHODS, ordered=True
    )
    selected = selected.sort_values(["model", "approach"])
    expected = {
        (model, method)
        for model in PERFORMANCE_MODELS
        for method in PERFORMANCE_METHODS
    }
    actual = {
        (str(row.model), str(row.approach))
        for row in selected.itertuples(index=False)
    }
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Missing overall model-method statistics: {missing}")
    return selected


def build_performance_table(statistics_path, output_dir):
    selected = select_performance_rows(statistics_path)
    csv_rows = selected[PERFORMANCE_COLUMNS].copy()
    csv_rows["model"] = csv_rows["model"].map(str)
    csv_rows["approach"] = csv_rows["approach"].map(METHOD_DISPLAY_NAMES)
    csv_rows.to_csv(output_dir / "Performance_comparison.csv", index=False)
    row_end = " " + "\\\\"
    lines = [
        r"\begin{tabular}{ll*{5}{c}c*{5}{c}c*{5}{c}c*{3}{c}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Method}} & \multicolumn{5}{c}{\textbf{File}} & & \multicolumn{5}{c}{\textbf{Function}} & & \multicolumn{5}{c}{\textbf{Line}} & & \multicolumn{3}{c}{\textbf{Cost}}" + row_end,
        r"\cline{3-7}\cline{9-13}\cline{15-19}\cline{21-23}",
        r"\noalign{\vskip 2pt}",
        r" & & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} & \textbf{MAP} & \textbf{MRR} & & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} & \textbf{MAP} & \textbf{MRR} & & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} & \textbf{MAP} & \textbf{MRR} & & \textbf{In (k)} & \textbf{Out (k)} & \textbf{\$}" + row_end,
        r"\midrule",
    ]
    for model_index, model in enumerate(PERFORMANCE_MODELS):
        group = selected[selected["model"] == model]
        maxima = {
            column: group[column].max(skipna=True)
            for column in PERFORMANCE_HIGHER_IS_BETTER
        }
        for row_index, (_, row) in enumerate(group.iterrows()):
            model_cell = (
                rf"\multirow{{{len(group)}}}{{*}}{{{MODEL_LATEX_NAMES[model]}}}"
                if row_index == 0
                else ""
            )
            metric_values = [
                bold_if_best(row[column], maxima[column], format_score(row[column]))
                for column in PERFORMANCE_HIGHER_IS_BETTER
            ]
            values = (
                metric_values[:5]
                + [""]
                + metric_values[5:10]
                + [""]
                + metric_values[10:15]
                + [""]
                + [
                    format_tokens(row["Input Tokens"]),
                    format_tokens(row["Output Tokens"]),
                    format_cost(row["$Cost"]),
                ]
            )
            lines.append(
                f"{model_cell} & {METHOD_LATEX[str(row['approach'])]} & "
                + " & ".join(values)
                + row_end
            )
        if model_index != len(PERFORMANCE_MODELS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "Performance_comparison.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return csv_rows.to_dict("records")


def parse_defect_paths(value):
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return list(parsed)


def build_dataset_statistics(classification_path, output_dir):
    classification = pd.read_csv(classification_path)
    category_rows = (
        classification.groupby(["benchmark", "defect_category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["benchmark", "defect_category"])
    )
    category_rows.to_csv(output_dir / "01classification.csv", index=False)
    file_counts = classification["defect_paths"].map(parse_defect_paths).map(len)
    statistics = [
        {
            "instances": int(len(file_counts)),
            "mean": file_counts.mean(),
            "maximum": file_counts.max(),
            "minimum": file_counts.min(),
            "variance": file_counts.var(),
            "standard_deviation": file_counts.std(),
        }
    ]
    write_rows(output_dir / "dataset_file_count_statistics.csv", list(statistics[0]), statistics)
    return category_rows.to_dict("records"), statistics


def load_or_create_ablation_summary(ablation_summary_path, ablation_results_root, output_dir, model):
    if ablation_summary_path.is_file():
        return pd.read_csv(ablation_summary_path)
    from paper_results_ablation import MODELS as ABLATION_MODELS
    from paper_results_ablation import calculate_summaries, make_rows

    if model not in ABLATION_MODELS:
        raise ValueError(f"Unsupported ablation model: {model}")
    logger.warning(f"Ablation summary is missing; extracting {model} from {ablation_results_root}")
    raw_rows = make_rows(ablation_results_root, [model], None)
    summary_rows = calculate_summaries(raw_rows)
    ablation_output_dir = ablation_summary_path.parent
    ablation_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(ablation_output_dir / "raw.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(ablation_summary_path, index=False)
    logger.info(f"Ablation raw rows: {len(raw_rows)}, first two: {raw_rows[:2]}")
    return pd.DataFrame(summary_rows)


def build_ablation_table(summary, output_dir, model):
    selected = summary[
        (summary["scope"] == "overall")
        & (summary["scope_value"] == "all")
        & (summary["model"] == model)
        & (summary["variant"].isin(ABLATION_VARIANTS))
    ].copy()
    expected = set(ABLATION_VARIANTS)
    missing = sorted(expected - set(selected["variant"]))
    if missing:
        raise ValueError(f"Missing ablation variants for {model}: {missing}")
    selected["variant"] = pd.Categorical(selected["variant"], categories=ABLATION_VARIANTS, ordered=True)
    selected = selected.sort_values("variant")
    metric_names = ["Top-1", "Top-3", "Top-5", "MAP", "MRR"]
    fields = ["model", "variant", *[f"{level} {metric}" for level in ("File", "Function", "Line") for metric in metric_names]]
    selected[fields].to_csv(output_dir / "ablation.csv", index=False)
    metric_columns = fields[2:]
    maxima = {column: selected[column].max(skipna=True) for column in metric_columns}
    lines = [
        "% Saved-run posthoc evidence and validator-stage comparison; not end-to-end component removal.",
        r"\begin{tabular}{lrrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Approach & \multicolumn{5}{c}{File} & \multicolumn{5}{c}{Function} & \multicolumn{5}{c}{Line} \\",
        " & " + " & ".join(metric_names * 3) + r" \\",
        r"\midrule",
    ]
    for _, row in selected.iterrows():
        if str(row["variant"]) == "full":
            lines.append(r"\midrule")
        values = [bold_if_best(row[column], maxima[column], format_score(row[column])) for column in metric_columns]
        name = ABLATION_NAMES[str(row["variant"])]
        if str(row["variant"]) == "full":
            name = rf"\textbf{{{name}}}"
        lines.append(f"{name} & " + " & ".join(values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected[fields].to_dict("records")


def main():
    parser = argparse.ArgumentParser(description="Generate all paper tables and their statistics")
    parser.add_argument(
        "--statistics",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_results" / "statistics.csv",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "classification_agreed.csv",
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_results_ablation" / "summary.csv",
    )
    parser.add_argument(
        "--ablation-results-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "results_main",
    )
    parser.add_argument("--ablation-model", choices=MODELS, default="kimi-k2.5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_results",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    approach_rows = build_approach_table(output_dir)
    performance_rows = build_performance_table(args.statistics.expanduser().resolve(), output_dir)
    category_rows, file_statistics = build_dataset_statistics(args.classification.expanduser().resolve(), output_dir)
    ablation_summary = load_or_create_ablation_summary(
        args.ablation_summary.expanduser().resolve(),
        args.ablation_results_root.expanduser().resolve(),
        output_dir,
        args.ablation_model,
    )
    ablation_rows = build_ablation_table(ablation_summary, output_dir, args.ablation_model)
    logger.info(f"Approach rows: {len(approach_rows)}, first two: {approach_rows[:2]}")
    logger.info(f"Performance rows: {len(performance_rows)}, first two: {performance_rows[:2]}")
    logger.info(f"Category rows: {len(category_rows)}, first two: {category_rows[:2]}")
    logger.info(f"Dataset file statistics: {file_statistics[:2]}")
    logger.info(f"Ablation rows: {len(ablation_rows)}, first two: {ablation_rows[:2]}")
    logger.info(f"Paper tables written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

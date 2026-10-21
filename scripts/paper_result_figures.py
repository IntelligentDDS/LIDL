import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from venn import venn4, venn5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = PROJECT_ROOT / "scripts" / "fonts"
TIMES_NEW_ROMAN_FILES = [
    FONT_DIR / "Times New Roman.ttf",
    FONT_DIR / "Times New Roman Bold.ttf",
    FONT_DIR / "Times New Roman Italic.ttf",
    FONT_DIR / "Times New Roman Bold Italic.ttf",
]
CATEGORY_ORDER = [
    "Prompt and Context Management",
    "LLM Interface Management",
    "Tool Integration Management",
    "LLM System Management",
]
CATEGORY_NAMES = {
    "Prompt and Context Management": "Prompt/Context",
    "LLM Interface Management": "LLM Interface",
    "Tool Integration Management": "Tool Integration",
    "LLM System Management": "LLM System",
}
CATEGORY_COLORS = ["#71B7ED", "#F2B56F", "#84C3B7", "#F57C6E"]
CATEGORY_METHODS = [
    "SWE-agent",
    "Agentless",
    "AutoCodeRover",
    "SWE-agent*",
    "Agentless*",
    "LocAgent",
    "Codex",
    "LIDL",
]
METHOD_COLORS = {
    "SWE-agent": "#8EC5E8",
    "Agentless": "#F3C178",
    "AutoCodeRover": "#93CFC1",
    "LocAgent": "#EAB6D1",
    "SWE-agent*": "#B6A0D8",
    "Agentless*": "#D9A38F",
    "Codex": "#A9C97E",
    "LIDL": "#F58A7E",
}
OVERLAP_METHODS = CATEGORY_METHODS


def configure_plotting():
    missing_fonts = [path for path in TIMES_NEW_ROMAN_FILES if not path.is_file()]
    if missing_fonts:
        raise FileNotFoundError(f"Missing Times New Roman font files: {missing_fonts}")
    for font_path in TIMES_NEW_ROMAN_FILES:
        font_manager.fontManager.addfont(font_path)
    logger.info(f"Registered Times New Roman fonts: {TIMES_NEW_ROMAN_FILES[:2]}")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["font.size"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["axes.grid"] = False


def save_figure(fig, output_base):
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".png"), format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_json_list(value):
    if pd.isna(value):
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    return parsed if isinstance(parsed, list) else []


def top_k_file_hit(row, cutoff=3):
    gold = set(parse_json_list(row["label_files"]))
    predictions = parse_json_list(row["localized_files"])
    return bool(gold.intersection(predictions[:cutoff]))


def parse_function_list(value):
    if pd.isna(value):
        return []
    items = json.loads(value) if isinstance(value, str) else value
    return list(dict.fromkeys((item["path"], item["name"]) for item in items))


def parse_line_list(value):
    if pd.isna(value):
        return []
    items = json.loads(value) if isinstance(value, str) else value
    return list(dict.fromkeys((item["path"], item["line"]) for item in items))


def rank_metrics(rows):
    metrics = []
    for row in rows.itertuples(index=False):
        file_gold = list(dict.fromkeys(parse_json_list(row.label_files)))
        file_predictions = list(dict.fromkeys(parse_json_list(row.localized_files)))
        function_gold = parse_function_list(row.label_functions)
        function_predictions = parse_function_list(row.localized_functions)
        line_gold_by_path = json.loads(row.label_lines or "{}")
        line_gold = list(
            dict.fromkeys(
                (path, line)
                for path, lines in line_gold_by_path.items()
                for line in lines
            )
        )
        line_predictions = parse_line_list(row.localized_ranked_lines)
        metrics.append(
            {
                "File Top-3": float(bool(set(file_gold).intersection(file_predictions[:3]))),
                "Function Top-3": float(bool(set(function_gold).intersection(function_predictions[:3]))),
                "Line Top-3": float(bool(set(line_gold).intersection(line_predictions[:3]))),
            }
        )
    return pd.DataFrame(metrics)


def draw_category_distribution(classification, output_dir):
    category_counts = classification["defect_category"].value_counts()
    counts = [int(category_counts.get(category, 0)) for category in CATEGORY_ORDER]
    labels = [CATEGORY_NAMES[category] for category in CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    wedges, _, autotexts = ax.pie(
        counts,
        colors=CATEGORY_COLORS,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9, "fontweight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)
    ax.legend(
        wedges,
        labels,
        title="Defect Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9,
        frameon=True,
        framealpha=0.7,
        handlelength=1.6,
        handleheight=0.8,
        handletextpad=0.2,
        columnspacing=0.9,
        borderpad=0.4,
        borderaxespad=0.5,
    )
    output_base = output_dir / "01defect_category_counts"
    save_figure(fig, output_base)
    rows = pd.DataFrame(
        {
            "defect_category": CATEGORY_ORDER,
            "display_name": labels,
            "count": counts,
            "fraction": np.array(counts) / sum(counts),
        }
    )
    rows.to_csv(output_base.with_suffix(".csv"), index=False)
    return rows.to_dict("records")


def calculate_category_metrics(raw, model):
    selected = raw[
        (raw["model"] == model)
        & (raw["approach"].isin(CATEGORY_METHODS))
        & (raw["defect_category"].isin(CATEGORY_ORDER))
    ].copy()
    rows = []
    for category in CATEGORY_ORDER:
        for method in CATEGORY_METHODS:
            group = selected[(selected["defect_category"] == category) & (selected["approach"] == method)]
            if group.empty:
                raise ValueError(f"Missing category results for {model} {category} {method}")
            values = rank_metrics(group)
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "method": method,
                    "instances": len(group),
                    **{column: values[column].mean() for column in values},
                }
            )
    return pd.DataFrame(rows)


def style_metric_axis(ax, label):
    ax.set_ylabel(label, fontweight="bold", fontsize=8)
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)


def draw_category_comparison(raw, output_dir, model):
    metrics = calculate_category_metrics(raw, model)
    output_base = output_dir / "8_defect_category_methods_comparison_combined"
    metrics.to_csv(output_base.with_suffix(".csv"), index=False)
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 4.2), sharex=True)
    bar_width = 0.165
    x = np.arange(len(CATEGORY_ORDER)) * 1.6
    metric_specs = [
        ("File Top-3", "File Top-3"),
        ("Function Top-3", "Function Top-3"),
        ("Line Top-3", "Line Top-3"),
    ]
    for ax, (column, label) in zip(axes, metric_specs):
        values_by_method = {}
        for method_index, method in enumerate(CATEGORY_METHODS):
            method_data = metrics[metrics["method"] == method].set_index("category").reindex(CATEGORY_ORDER)
            values = method_data[column].to_numpy()
            values_by_method[method] = values
            positions = x + bar_width * (method_index - (len(CATEGORY_METHODS) - 1) / 2)
            ax.bar(
                positions,
                values,
                width=bar_width,
                color=METHOD_COLORS[method],
                edgecolor="#222222",
                linewidth=0.6,
                label=method,
            )
        for category_index in range(len(CATEGORY_ORDER)):
            for method_index, method in enumerate(CATEGORY_METHODS):
                value = values_by_method[method][category_index]
                position = x[category_index] + bar_width * (method_index - (len(CATEGORY_METHODS) - 1) / 2)
                ax.text(
                    position,
                    value + 0.008,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=3.8,
                )
        style_metric_axis(ax, label)
        ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([CATEGORY_NAMES[category] for category in CATEGORY_ORDER], fontweight="bold", fontsize=8)
    legend_elements = [
        Patch(facecolor=METHOD_COLORS[method], edgecolor="#222222", linewidth=0.6, label=method)
        for method in CATEGORY_METHODS
    ]
    axes[0].legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=len(CATEGORY_METHODS),
        fontsize=5.6,
        frameon=True,
        framealpha=0.7,
        handlelength=0.8,
        handleheight=0.55,
        handletextpad=0.2,
        columnspacing=0.5,
        borderpad=0.4,
        borderaxespad=0.5,
    )
    fig.subplots_adjust(hspace=0.38)
    save_figure(fig, output_base)
    return metrics.to_dict("records")


def successful_instances(raw, model, method, cutoff=3):
    selected = raw[(raw["model"] == model) & (raw["approach"] == method)]
    if selected.empty:
        raise ValueError(f"Missing Venn results for {model} {method}")
    return set(selected[selected.apply(top_k_file_hit, axis=1, cutoff=cutoff)]["instance_id"])


def exact_venn_regions(methods, method_sets):
    regions = {}
    for size in range(1, len(methods) + 1):
        for combination in combinations(range(len(methods)), size):
            binary = "".join("1" if index in combination else "0" for index in range(len(methods)))
            instances = set.intersection(*(method_sets[methods[index]] for index in combination))
            for index in range(len(methods)):
                if index not in combination:
                    instances -= method_sets[methods[index]]
            regions[binary] = len(instances)
    return regions


def draw_overlap(raw, output_dir, model):
    method_sets = {
        method: successful_instances(raw, model, method, cutoff=3)
        for method in OVERLAP_METHODS
    }
    panels = [
        (
            "Non-graph",
            ["SWE-agent", "Agentless", "AutoCodeRover", "Codex", "LIDL"],
            venn5,
        ),
        (
            "Graph-based",
            ["SWE-agent*", "Agentless*", "LocAgent", "LIDL"],
            venn4,
        ),
    ]
    panel_regions = {
        title: exact_venn_regions(methods, method_sets)
        for title, methods, _ in panels
    }
    def rgba(method):
        red, green, blue = plt.matplotlib.colors.to_rgb(METHOD_COLORS[method])
        return [red, green, blue, 0.60]

    fig, axes = plt.subplots(1, 2, figsize=(7.10, 3.55), dpi=100)
    for ax, (title, methods, venn_function) in zip(axes, panels):
        panel_colors = [rgba(method) for method in methods]
        labels = {
            combination: str(count)
            for combination, count in panel_regions[title].items()
        }
        venn_function(
            labels=labels,
            names=methods,
            colors=panel_colors,
            fontsize=12,
            ax=ax,
        )
        method_names = set(methods)
        for text in list(ax.texts):
            if text.get_text() in method_names:
                text.remove()
            elif text.get_text().strip():
                text.set_fontweight("bold")
        legend = ax.get_legend()
        if legend:
            legend.remove()
    legend_groups = [
        (0.28, panels[0][1]),
        (0.72, panels[1][1]),
    ]
    for center, methods in legend_groups:
        handles = [
            Patch(
                facecolor=METHOD_COLORS[method],
                edgecolor=METHOD_COLORS[method],
                alpha=0.60,
                label=method,
            )
            for method in methods
        ]
        legend = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(center, 0.9),
            ncol=len(methods),
            handlelength=0.7,
            handleheight=0.45,
            handletextpad=0.18,
            columnspacing=0.45,
            labelspacing=0.30,
            borderpad=0.24,
            prop={"family": "Times New Roman", "size": 8.5, "weight": "normal"},
            framealpha=0.97,
            fancybox=True,
        )
        legend.get_frame().set_edgecolor("#bdbdbd")
    fig.text(0.28, 0.07, "Non-graph", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    fig.text(0.72, 0.07, "Graph-based", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.10, wspace=-0.2)

    output_base = output_dir / "venn_method_overlap"
    save_figure(fig, output_base)
    rows = [
        {
            "panel": title,
            "methods": ", ".join(methods),
            "combination": combination,
            "count": count,
        }
        for title, methods, _ in panels
        for combination, count in panel_regions[title].items()
    ]
    pd.DataFrame(rows).to_csv(output_base.with_suffix(".csv"), index=False)
    return rows


def add_box(ax, x, y, width, height, text, facecolor, edgecolor="#444444", fontsize=8, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
    )


def add_arrow(ax, start, end, color="#555555", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def draw_reasoning_comparison(output_dir):
    rows = [
        {
            "lane": "Code-centric baselines",
            "step": 1,
            "node": "Runtime trace",
            "description": "Follow the observed execution path",
            "classification": "signal",
        },
        {
            "lane": "Code-centric baselines",
            "step": 2,
            "node": "openai_client.py",
            "description": "Execution stalls at the API client",
            "classification": "symptom",
        },
        {
            "lane": "Code-centric baselines",
            "step": 3,
            "node": "Symptom file",
            "description": "Incorrect localization",
            "classification": "incorrect",
        },
        {
            "lane": "LIDL",
            "step": 1,
            "node": "Runtime + semantic evidence",
            "description": "Fuse execution signals and LLM-specific annotations",
            "classification": "signal",
        },
        {
            "lane": "LIDL",
            "step": 2,
            "node": "Semantic link",
            "description": "Bridge API invocation and prompt construction",
            "classification": "semantic",
        },
        {
            "lane": "LIDL",
            "step": 3,
            "node": "Counterfactual validation",
            "description": "Ask whether changing the candidate removes the defect",
            "classification": "validation",
        },
        {
            "lane": "LIDL",
            "step": 4,
            "node": "prompts.py",
            "description": "True root-cause file",
            "classification": "root cause",
        },
    ]
    output_base = output_dir / "5reasoning_comparison"
    pd.DataFrame(rows).to_csv(output_base.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.79, "Code-centric\nbaselines", ha="left", va="center", fontsize=10, fontweight="bold", color="#5A5A5A")
    ax.text(0.02, 0.28, "LIDL", ha="left", va="center", fontsize=11, fontweight="bold", color="#C94F45")
    baseline_positions = [(0.20, 0.70), (0.48, 0.70), (0.76, 0.70)]
    baseline_texts = ["Runtime trace", "openai_client.py\nAPI invocation layer", "Symptom file\nIncorrect localization"]
    baseline_colors = ["#EAF3FA", "#FFF0D8", "#FADBD8"]
    for (x, y), text, color in zip(baseline_positions, baseline_texts, baseline_colors):
        add_box(ax, x, y, 0.20, 0.17, text, color, bold=text.startswith(("openai", "Symptom")))
    for left, right in zip(baseline_positions, baseline_positions[1:]):
        add_arrow(ax, (left[0] + 0.20, left[1] + 0.085), (right[0], right[1] + 0.085))
    lidl_positions = [(0.16, 0.16), (0.39, 0.16), (0.62, 0.16), (0.84, 0.16)]
    lidl_widths = [0.18, 0.18, 0.18, 0.14]
    lidl_texts = ["Runtime + semantic\nevidence", "LLM-specific\nsemantic link", "Counterfactual\nvalidation", "prompts.py\nRoot cause"]
    lidl_colors = ["#EAF3FA", "#E6F3EF", "#EEE5F5", "#DDF0DC"]
    for (x, y), width, text, color in zip(lidl_positions, lidl_widths, lidl_texts, lidl_colors):
        add_box(ax, x, y, width, 0.18, text, color, bold=text.startswith(("Counterfactual", "prompts")))
    for index in range(len(lidl_positions) - 1):
        add_arrow(
            ax,
            (lidl_positions[index][0] + lidl_widths[index], lidl_positions[index][1] + 0.09),
            (lidl_positions[index + 1][0], lidl_positions[index + 1][1] + 0.09),
            color="#5A7D6E",
        )
    add_arrow(ax, (0.58, 0.70), (0.47, 0.34), color="#C94F45", connectionstyle="arc3,rad=0.18")
    ax.text(0.52, 0.49, "No direct call edge", ha="center", va="center", fontsize=8, color="#C94F45", fontweight="bold")
    ax.plot([0.12, 0.98], [0.56, 0.56], color="#CCCCCC", linewidth=1)
    save_figure(fig, output_base)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures and their statistics")
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_results" / "raw.csv",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "classification_agreed.csv",
    )
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_results",
    )
    args = parser.parse_args()
    configure_plotting()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw.expanduser().resolve())
    classification = pd.read_csv(args.classification.expanduser().resolve())
    category_rows = draw_category_distribution(classification, output_dir)
    comparison_rows = draw_category_comparison(raw, output_dir, args.model)
    overlap_rows = draw_overlap(raw, output_dir, args.model)
    reasoning_rows = draw_reasoning_comparison(output_dir)
    # logger.info(f"Category distribution rows: {len(category_rows)}, first two: {category_rows[:2]}")
    # logger.info(f"Category comparison rows: {len(comparison_rows)}, first two: {comparison_rows[:2]}")
    # logger.info(f"Overlap rows: {len(overlap_rows)}, first two: {overlap_rows[:2]}")
    # logger.info(f"Reasoning rows: {len(reasoning_rows)}, first two: {reasoning_rows[:2]}")
    # logger.info(f"Paper figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

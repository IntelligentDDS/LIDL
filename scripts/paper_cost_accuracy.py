import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, MultipleLocator

from paper_result_figures import METHOD_COLORS, configure_plotting, save_figure
from paper_result_tables import MODEL_DISPLAY_NAMES, PERFORMANCE_METHODS, select_performance_rows

ROOT = Path(__file__).resolve().parents[1]
MODEL_MARKERS = {
    "gemma-4-31b": "o",
    "glm-5-2-260617": "s",
    "deepseek-v4-flash-ga-260731": "^",
    "gpt-5.6-terra": "D",
}
CAPTION = (
    "Cost and localization accuracy across methods and backbone LLMs. "
    "Each point represents a method and model combination. "
    "The horizontal axis shows average cost per instance in USD, using a logarithmic scale. "
    "The vertical axis shows line Top-3. Colors indicate methods, and marker shapes indicate "
    "backbone LLMs. Points closer to the upper left have lower cost and higher accuracy."
)


def load_points(statistics, limit):
    rows = select_performance_rows(statistics)
    if rows.duplicated(["model", "approach"]).any() or len(rows) != 32:
        raise ValueError("Expected exactly 32 unique method-model combinations")
    if not np.isfinite(rows[["$Cost", "Line Top-3"]].to_numpy()).all():
        raise ValueError("Missing or nonfinite cost or Line Top-3")
    if (rows["$Cost"] <= 0).any() or not rows["Line Top-3"].between(0, 1).all():
        raise ValueError("Log-scale costs must be positive and Line Top-3 must lie in [0, 1]")
    if limit is not None:
        logger.warning(f"Validation only: plotting the first {limit} real method-model records in a separate directory")
        rows = rows.head(limit)
    points = rows[["model", "approach", "Line Top-3", "$Cost", "Line Instances", "token_instances", "cost_instances"]].copy()
    logger.info(f"First two plotting records: {points.head(2).to_dict('records')}")
    return points


def draw_cost_accuracy(points, output_dir):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.16, top=0.97)
    for row in points.to_dict("records"):
        ax.scatter(
            row["$Cost"], row["Line Top-3"],
            color=METHOD_COLORS[row["approach"]], marker=MODEL_MARKERS[row["model"]],
            s=68, edgecolors="#444444", linewidths=0.55, zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(points["$Cost"].min() / 1.6, points["$Cost"].max() * 1.6)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Cost (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Line Top-3", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    methods = [
        Line2D([], [], color=METHOD_COLORS[method], linewidth=7, solid_capstyle="butt",
               label=method.replace("*", r"$^*$"))
        for method in PERFORMANCE_METHODS
    ]
    models = [
        Line2D([], [], color="#444444", marker=marker, linestyle="none",
               markerfacecolor="white", markeredgewidth=0.8, markersize=7,
               label="GLM-5.2" if model == "glm-5-2-260617" else MODEL_DISPLAY_NAMES[model])
        for model, marker in MODEL_MARKERS.items()
    ]
    method_legend = ax.legend(handles=methods, title="Method", loc="upper center",
                              bbox_to_anchor=(1.20, 1.02), frameon=False, alignment="center",
                              fontsize=10, title_fontsize=12, borderaxespad=0,
                              handlelength=1.2, handletextpad=0.4, labelspacing=0.3, borderpad=0)
    method_legend.get_title().set_fontweight("bold")
    ax.add_artist(method_legend)
    model_legend = ax.legend(handles=models, title="LLM", loc="upper center",
                            bbox_to_anchor=(1.20, 0.40), frameon=False, alignment="center",
                            fontsize=10, title_fontsize=12, borderaxespad=0,
                            handlelength=1.2, handletextpad=0.4, labelspacing=0.3, borderpad=0)
    model_legend.get_title().set_fontweight("bold")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_width = max(legend.get_window_extent(renderer).width for legend in (method_legend, model_legend))
    axes_box = ax.get_window_extent(renderer)
    center = 1 + 0.005 + legend_width / (2 * axes_box.width)
    method_legend.set_bbox_to_anchor((center, 1.0))
    group_gap = 8 * fig.dpi / 72
    model_top = 1.0 - (method_legend.get_window_extent(renderer).height + group_gap) / axes_box.height
    model_legend.set_bbox_to_anchor((center, model_top))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "cost_accuracy"
    points.to_csv(output_base.with_suffix(".csv"), index=False)
    output_base.with_suffix(".tex").write_text("\\caption{" + CAPTION + "}\n", encoding="utf-8")
    save_figure(fig, output_base)
    logger.info(f"Saved {len(points)} points and PDF/PNG/CSV/caption to {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Plot cost versus line Top-3 for the paper's 32 method-model combinations")
    parser.add_argument("--statistics", type=Path, default=ROOT / "results/paper_results/statistics.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_results")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 32:
        parser.error("--limit must be between 1 and 32")
    output_dir = args.output_dir if args.limit is None else args.output_dir / f"cost_accuracy_validation_limit{args.limit}"
    draw_cost_accuracy(load_points(args.statistics, args.limit), output_dir)


if __name__ == "__main__":
    main()

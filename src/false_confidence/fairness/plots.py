"""Visualization functions for fairness analysis.

Each function takes data + an Report and saves figures via report.figure().
Called by analyze.py.
"""

from __future__ import annotations

import polars as pl
import seaborn as sns

from false_confidence.report import Report

sns.set_theme(style="whitegrid", palette="muted")


def violin_by_group(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    report: Report,
    *,
    title: str | None = None,
    fig_name: str | None = None,
) -> None:
    """Violin plot of a score by demographic group.

    Pass an explicit ``fig_name`` to disambiguate plots that share a
    ``group_col`` (e.g. the race_wb/wbo/wn strategies all map to
    ``race_group``, and each ruler reuses the same columns). Without it,
    those plots overwrite each other on disk.
    """
    pdf = df.select([score_col, group_col]).to_pandas()
    if fig_name is None:
        fig_name = f"violin_{score_col}_by_{group_col}"

    with report.figure(fig_name, figsize=(8, 5)) as fig:
        ax = fig.gca()
        sns.violinplot(data=pdf, x=group_col, y=score_col, ax=ax, inner="quart")
        ax.set_xlabel(group_col.replace("_", " ").title())
        ax.set_ylabel(score_col.replace("_", " ").title())
        ax.set_title(title or f"{score_col} by {group_col}")


def dir_bar_chart(
    gaps: list[dict],
    labels: list[str],
    report: Report,
    *,
    fig_name: str = "dir_bar_chart",
) -> None:
    """Horizontal bar chart of DIR values with four-fifths rule threshold."""
    dir_values = [g["dir"] for g in gaps]

    with report.figure(fig_name, figsize=(8, max(4, len(labels) * 0.8))) as fig:
        ax = fig.gca()
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, dir_values, height=0.5)

        for bar, val in zip(bars, dir_values):
            color = "#d9534f" if val < 0.8 else "#5cb85c"
            bar.set_color(color)

        ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1, label="Four-fifths rule (0.8)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Disparate Impact Ratio")
        ax.set_xlim(0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Disparate Impact Ratio by Demographic Grouping")


def cross_ruler_dir(
    ruler_gaps: dict[str, list[dict]],
    grouping_labels: list[str],
    report: Report,
    *,
    fig_name: str = "cross_ruler_dir",
) -> None:
    """Grouped bar chart comparing DIR across rulers."""
    import numpy as np

    rulers = list(ruler_gaps.keys())
    n_groups = len(grouping_labels)
    n_rulers = len(rulers)
    x = np.arange(n_groups)
    width = 0.8 / n_rulers

    with report.figure(fig_name, figsize=(max(8, n_groups * 2), 5)) as fig:
        ax = fig.gca()
        for i, ruler in enumerate(rulers):
            dir_values = [g["dir"] for g in ruler_gaps[ruler]]
            offset = (i - n_rulers / 2 + 0.5) * width
            ax.bar(x + offset, dir_values, width, label=ruler)

        ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(grouping_labels, rotation=45, ha="right")
        ax.set_ylabel("Disparate Impact Ratio")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("DIR Comparison Across Rulers")


def bootstrap_forest(
    ci_results: list[dict],
    labels: list[str],
    report: Report,
    *,
    fig_name: str = "bootstrap_forest",
) -> None:
    """Forest plot: point estimate + CI error bars for DIR by group."""
    with report.figure(fig_name, figsize=(8, max(4, len(labels) * 0.6))) as fig:
        ax = fig.gca()
        y_pos = range(len(labels))
        points = [r["point_estimate"] for r in ci_results]
        ci_lows = [r["ci_low"] for r in ci_results]
        ci_highs = [r["ci_high"] for r in ci_results]

        errors = [
            [p - lo for p, lo in zip(points, ci_lows)],
            [hi - p for p, hi in zip(points, ci_highs)],
        ]

        ax.errorbar(points, y_pos, xerr=errors, fmt="o", capsize=4, color="steelblue")
        ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Disparate Impact Ratio")
        ax.set_title("DIR with Bootstrap 95% CI")

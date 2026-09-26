"""Figure 1: the variance collapse behind the E2 verdict flip.

Macro Dice by age group for the same M_mix predictions on the 76 gold-test
images, scored against the gold ruler (expert labels, left) and the silver
ruler (M_gold's predictions, right). Boxes are IQRs, whiskers +/- 1 SD; the
FDR-corrected Kruskal-Wallis p-value is read from the analysis output.

Usage:
    uv run python scripts/plot_age_trend.py [--run outputs/fairness/fairness_biased_ruler/<timestamp>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

RUNS = Path("outputs/fairness/fairness_biased_ruler")
OUT = Path("outputs/figures/age_trend_gold_vs_silver.png")
AGE_ORDER = ["<40", "40-60", "60+"]


def latest_run() -> Path:
    runs = sorted(p for p in RUNS.glob("*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No analysis runs in {RUNS}; run scripts/05_audit.sh first.")
    return runs[-1]


def load_summary(run: Path, ruler: str) -> dict[str, dict]:
    df = pl.read_csv(run / f"summary_{ruler}__dice_macro__age_3bin.csv")
    return {row["group"]: row for row in df.iter_rows(named=True)}


def load_fdr_p(run: Path, ruler: str) -> float:
    """FDR-corrected Kruskal-Wallis p for macro Dice ~ age (3 bins)."""
    fdr = pl.read_csv(run / f"fdr_{ruler}.csv")
    row = fdr.filter(pl.col("test") == f"{ruler}__dice_macro__age_3bin")
    return float(row["p_fdr"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 1 (age trend, gold vs. silver ruler)")
    parser.add_argument("--run", type=Path, default=None, help="Analysis run directory")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run = args.run or latest_run()
    print(f"Reading {run}")

    sns.set_theme(style="whitegrid", palette="muted")
    muted = sns.color_palette("muted")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    panels = [
        ("Gold ruler (expert labels)", "gold", muted[0]),  # blue
        (
            "Silver ruler ($M_{\\mathrm{gold}}$ predictions)",
            "silver",
            muted[1],
        ),  # orange
    ]

    for ax, (title, ruler, color) in zip(axes, panels):
        s = load_summary(run, ruler)
        xs = list(range(len(AGE_ORDER)))
        medians = []

        for x, g in zip(xs, AGE_ORDER):
            r = s[g]
            q25, med, q75 = r["q25"], r["median"], r["q75"]
            mean, std = r["mean"], r["std"]
            medians.append(med)

            # IQR box
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.25, q25),
                    0.50,
                    q75 - q25,
                    facecolor=color,
                    alpha=0.25,
                    edgecolor=color,
                    linewidth=1.2,
                )
            )
            # Median line
            ax.plot(
                [x - 0.25, x + 0.25],
                [med, med],
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=5,
            )
            # ±1 std whiskers
            ax.plot(
                [x, x],
                [mean - std, mean + std],
                color=color,
                linewidth=1.0,
                alpha=0.7,
                zorder=3,
            )
            ax.plot(
                [x - 0.06, x + 0.06],
                [mean - std, mean - std],
                color=color,
                linewidth=1.0,
            )
            ax.plot(
                [x - 0.06, x + 0.06],
                [mean + std, mean + std],
                color=color,
                linewidth=1.0,
            )
            # Mean marker
            ax.plot(
                x,
                mean,
                "o",
                color="white",
                markersize=5,
                markeredgecolor=color,
                markeredgewidth=1.2,
                zorder=6,
            )

        # Connect medians
        ax.plot(xs, medians, "--", color=color, linewidth=1.2, alpha=0.7, zorder=4)

        # Annotation: std and verdict
        p = load_fdr_p(run, ruler)
        stds = [s[g]["std"] for g in AGE_ORDER]
        std_range = f"std: {min(stds):.3f}–{max(stds):.3f}"
        sig_label = f"$p_{{\\mathrm{{fdr}}}} = {p:.3f}$"
        if p < 0.05:
            sig_label += " (significant)"
        else:
            sig_label += " (n.s.)"

        ax.text(
            0.97,
            0.05,
            f"{std_range}\n{sig_label}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(AGE_ORDER)
        ax.set_xlabel("Age group")
        ax.set_xlim(-0.6, len(AGE_ORDER) - 0.4)

    axes[0].set_ylabel("Macro Dice")
    axes[0].set_ylim(0.75, 1.0)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")

    # Console check
    for ruler in ("gold", "silver"):
        s = load_summary(run, ruler)
        line = "  ".join(f"{g}: med={s[g]['median']:.3f} std={s[g]['std']:.3f}" for g in AGE_ORDER)
        print(f"{ruler:6s} | {line}")


if __name__ == "__main__":
    main()

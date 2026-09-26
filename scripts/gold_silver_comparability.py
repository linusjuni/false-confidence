"""Gold vs. silver pool comparability (paper, Discussion: limitation iv).

Compares the gold (expert) and silver (machine-labelled) pools of the analysis
cohort (``splits/split_mix.tsv``) on sex, race, age, manufacturer and field
strength (chi-squared / Mann-Whitney with Benjamini-Hochberg correction). The
pools were split by enrollment order, not at random, and differ modestly in
race and age.

Usage:
    uv run python scripts/gold_silver_comparability.py

Writes outputs/eda/gold_silver_comparability.json and prints a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl

from false_confidence.data.groups import AgeStrategy, RaceStrategy, age, race
from false_confidence.data.loader import load_metadata
from false_confidence.data.schemas import Col
from false_confidence.splits import MIX, apply_splits
from false_confidence.stats import chi2_result, mann_whitney_result

SPLIT_VERSION = MIX
QUALITY = "annotation_quality"


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values, same family-wise correction the
    paper uses. Returns adjusted p-values in the original order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])  # ascending by raw p
    adj = [0.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):  # walk from largest p (rank n) down to 1
        idx = order[rank - 1]
        running_min = min(running_min, pvals[idx] * n / rank)
        adj[idx] = min(running_min, 1.0)
    return adj


def pool_percentages(ct: pd.DataFrame) -> dict:
    """Per-pool (row = gold/silver) percentage breakdown across categories."""
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    return {pool: pct.loc[pool].to_dict() for pool in pct.index}


def compare_categorical(df: pl.DataFrame, cat_col: str, label: str) -> dict:
    """Chi-squared independence of a categorical attribute vs. gold/silver."""
    pdf = df.select(QUALITY, cat_col).drop_nulls().to_pandas()
    ct = pd.crosstab(pdf[QUALITY], pdf[cat_col])
    res = chi2_result(ct)
    return {
        "attribute": label,
        "kind": "categorical",
        "counts": {pool: ct.loc[pool].to_dict() for pool in ct.index},
        "pool_pct": pool_percentages(ct),
        "test": res["test"],
        "stat": res["chi2"],
        "dof": res["dof"],
        "effect": {"name": "cramers_v", "value": res["cramers_v"]},
        "p": res["p"],
    }


def compare_age_continuous(df: pl.DataFrame) -> dict:
    """Mann-Whitney on raw age, gold vs. silver (nulls = confirmed >89, dropped)."""
    gold = df.filter(pl.col(QUALITY) == "gold")[Col.AGE].drop_nulls().to_list()
    silver = df.filter(pl.col(QUALITY) == "silver")[Col.AGE].drop_nulls().to_list()
    res = mann_whitney_result(gold, silver)
    return {
        "attribute": "Age (continuous, years)",
        "kind": "continuous",
        "median_gold": res["median_a"],
        "median_silver": res["median_b"],
        "n_gold": res["n_a"],
        "n_silver": res["n_b"],
        "test": res["test"],
        "stat": res["U"],
        "effect": {"name": "r_rb", "value": res["r_rb"]},
        "p": res["p"],
    }


def main() -> None:
    df = apply_splits(load_metadata(), SPLIT_VERSION)
    df = df.filter(pl.col(QUALITY).is_not_null())

    n_gold = df.filter(pl.col(QUALITY) == "gold").height
    n_silver = df.filter(pl.col(QUALITY) == "silver").height
    print(
        f"\nAnalysis cohort D ({SPLIT_VERSION}): "
        f"{df.height} exams = {n_gold} gold + {n_silver} silver\n"
    )

    results: list[dict] = []

    # Sex, manufacturer, field strength: straight categorical contingencies on
    # the full cohort.
    results.append(compare_categorical(df, str(Col.SEX), "Sex"))
    results.append(compare_categorical(df, str(Col.MANUFACTURER), "Manufacturer"))
    results.append(compare_categorical(df, str(Col.FIELD_STRENGTH), "Field strength"))

    # Race: each grouping is derived from the full df via the paper's registry,
    # which drops unmapped rows internally -- so it cannot corrupt the cohort df.
    df_wb = race[RaceStrategy.WHITE_VS_BLACK].apply(df, Col.RACE)
    results.append(compare_categorical(df_wb, "race_group", "Race (White vs. Black)"))
    df_wbo = race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER].apply(df, Col.RACE)
    results.append(compare_categorical(df_wbo, "race_group", "Race (White/Black/Other)"))

    # Age: continuous Mann-Whitney + 3-bin contingency (the paper's bins).
    results.append(compare_age_continuous(df))
    df_age = age[AgeStrategy.THREE_BINS].apply(df, Col.AGE)
    results.append(compare_categorical(df_age, "age_at_imaging_group", "Age (3-bin)"))

    # Family-wise FDR correction across all comparisons.
    adj = benjamini_hochberg([r["p"] for r in results])
    for r, q in zip(results, adj):
        r["p_fdr"] = q
        eff = abs(r["effect"]["value"])
        r["comparable"] = bool(q >= 0.05 and eff < 0.1)

    # --- readable summary --------------------------------------------------
    print(f"{'Attribute':<28}{'test':>8}{'effect':>14}{'p':>9}{'p_fdr':>9}  verdict")
    print("-" * 84)
    for r in results:
        eff = f"{r['effect']['name']}={r['effect']['value']:+.3f}"
        verdict = "comparable" if r["comparable"] else "** CHECK **"
        print(
            f"{r['attribute']:<28}{r['test'][:8]:>8}{eff:>14}"
            f"{r['p']:>9.3f}{r['p_fdr']:>9.3f}  {verdict}"
        )
    print()
    for r in results:
        if r["kind"] == "categorical":
            print(f"  {r['attribute']}:")
            for pool, pct in r["pool_pct"].items():
                cells = ", ".join(f"{k} {v}%" for k, v in pct.items())
                print(f"     {pool:<7} {cells}")
    print()

    all_ok = all(r["comparable"] for r in results)
    print(
        "=> Gold and silver pools are demographically + scanner-comparable."
        if all_ok
        else "=> At least one marginal differs -- inspect the flagged row(s)."
    )

    report = {
        "split_version": SPLIT_VERSION,
        "n_gold": n_gold,
        "n_silver": n_silver,
        "all_comparable": all_ok,
        "comparisons": results,
    }
    out = Path("outputs/eda/gold_silver_comparability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote audit JSON -> {out}\n")


if __name__ == "__main__":
    main()

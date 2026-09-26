"""Cohort composition of the working set D_0 (paper, Table 1).

Usage:
    uv run python scripts/cohort_composition.py

Prints exam-level counts (N_0 = 1,254) and writes
outputs/eda/cohort_composition.json. Percentages are of N_0.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from false_confidence.data.groups import AgeStrategy, age
from false_confidence.data.loader import load_metadata
from false_confidence.data.schemas import Col


def counts(df: pl.DataFrame, col: str) -> list[dict]:
    """Value counts for a categorical column, descending, with % of total."""
    n = df.height
    vc = (
        df.group_by(col)
        .len()
        .sort("len", descending=True)
        .with_columns((pl.col("len") / n * 100).round(1).alias("pct"))
    )
    return [
        {"value": row[col], "n": row["len"], "pct": row["pct"]} for row in vc.iter_rows(named=True)
    ]


def main() -> None:
    df = load_metadata()
    n0 = df.height
    print(
        f"\nWorking set D_0: N_0 = {n0} exams ({df[Col.PATIENT_ID].n_unique()} unique patients)\n"
    )

    report: dict = {"n0_exams": n0, "n_patients": df[Col.PATIENT_ID].n_unique()}

    # --- categorical breakdowns --------------------------------------------
    for label, col in [
        ("Sex", Col.SEX),
        ("Race", Col.RACE),
        ("Ethnicity", Col.ETHNICITY),
        ("Manufacturer", Col.MANUFACTURER),
        ("Field strength", Col.FIELD_STRENGTH),
    ]:
        rows = counts(df, str(col))
        report[str(col)] = rows
        n_missing = df[str(col)].null_count()
        print(f"== {label} ({col}) ==  [missing: {n_missing}]")
        for r in rows:
            print(f"   {str(r['value']):<42} {r['n']:>5}  ({r['pct']:>4}%)")
        print()

    # --- age: continuous summary + project 3-bin ---------------------------
    age_col = str(Col.AGE)
    age_stats = {
        "mean": df[age_col].mean(),
        "sd": df[age_col].std(),
        "median": df[age_col].median(),
        "min": df[age_col].min(),
        "max": df[age_col].max(),
        "n_missing": df[age_col].null_count(),
    }
    report["age_stats"] = age_stats
    print("== Age (years) ==")
    print(
        f"   mean {age_stats['mean']:.1f} +/- {age_stats['sd']:.1f}, "
        f"median {age_stats['median']}, range {age_stats['min']}-{age_stats['max']}, "
        f"missing {age_stats['n_missing']}"
    )

    binned = age[AgeStrategy.THREE_BINS].apply(df, Col.AGE)
    bin_rows = counts(binned, "age_at_imaging_group")
    report["age_three_bins"] = bin_rows
    print("   3-bin (<40 / 40-60 / 60+; null ages -> 60+):")
    for r in bin_rows:
        print(f"      {str(r['value']):<10} {r['n']:>5}  ({r['pct']:>4}%)")
    print()

    out = Path("outputs/eda/cohort_composition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote audit JSON -> {out}\n")


if __name__ == "__main__":
    main()

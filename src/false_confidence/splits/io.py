"""Reading and writing split files.

Split files are exam-level TSVs with four columns:

    patient_id  series_submitter_id  split  annotation_quality

``split`` is ``train`` / ``val`` / ``test`` and ``annotation_quality`` is
``gold`` (expert) or ``silver`` (machine-generated). Demographic strata are not
stored; :func:`load_splits` recomputes them from the MIDRC metadata.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from false_confidence.data.groups import AgeStrategy, RaceStrategy, age, race
from false_confidence.data.loader import load_metadata
from false_confidence.data.schemas import Col
from false_confidence.logger import get_logger
from false_confidence.settings import settings

logger = get_logger(__name__)

# Split versions used in the paper.
MIX = "split_mix"  # M_mix: all labels, sex-balanced (N = 1,142)
GOLD = "split_gold"  # M_gold: expert labels only, sex-balanced (N = 408)

SPLIT_COLUMNS = [Col.PATIENT_ID, Col.SERIES_SUBMITTER_ID, "split", "annotation_quality"]
STRATA_COLUMNS = ["race_bin", "age_bin", "sex_bin", "stratum"]


def add_strata(df: pl.DataFrame) -> pl.DataFrame:
    """Add race_bin, age_bin, sex_bin and the stratification key ``stratum``.

    ``df`` must carry the raw race, sex and age columns from ``load_metadata``.
    Strata are race (White/Black) x age (<40/40-60/60+) x sex; all other races
    share the single stratum ``Other``.
    """
    df = race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER].apply(df, Col.RACE)
    df = df.rename({f"{Col.RACE}_group": "race_bin"})
    df = age[AgeStrategy.THREE_BINS].apply(df, Col.AGE)
    df = df.rename({f"{Col.AGE}_group": "age_bin"})
    return df.with_columns(
        pl.col(Col.SEX).alias("sex_bin"),
        pl.when(pl.col("race_bin").is_in(["White", "Black"]))
        .then(pl.col("race_bin") + "_" + pl.col("age_bin") + "_" + pl.col(Col.SEX))
        .otherwise(pl.lit("Other"))
        .alias("stratum"),
    )


def save_splits(df: pl.DataFrame, version: str, out_dir: Path | None = None) -> None:
    """Write the four split columns to ``<out_dir or SPLITS_DIR>/<version>.tsv``."""
    out_dir = out_dir or settings.splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{version}.tsv"
    df.select(SPLIT_COLUMNS).write_csv(path, separator="\t")
    logger.success("Saved splits", path=str(path), rows=df.height)


def load_splits(version: str, *, with_strata: bool = True) -> pl.DataFrame:
    """Load ``SPLITS_DIR/<version>.tsv``, optionally joined with demographic strata.

    Row order of the file is preserved (sex balancing depends on it).
    """
    path = settings.splits_dir / f"{version}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found: {path}")
    df = pl.read_csv(path, separator="\t", columns=SPLIT_COLUMNS)
    if not with_strata:
        return df
    meta = add_strata(load_metadata()).select(Col.SERIES_SUBMITTER_ID, *STRATA_COLUMNS)
    return df.join(meta, on=Col.SERIES_SUBMITTER_ID, how="left", maintain_order="left")


def apply_splits(df: pl.DataFrame, version: str) -> pl.DataFrame:
    """Inner-join split assignment and provenance onto exam-level metadata."""
    splits = load_splits(version, with_strata=False).drop(Col.PATIENT_ID)
    return add_strata(df).join(splits, on=Col.SERIES_SUBMITTER_ID, how="inner")


def log_balance(df: pl.DataFrame) -> None:
    """Log exam counts per stratum x split."""
    balance = (
        df.group_by("stratum", "split")
        .len()
        .pivot(on="split", index="stratum", values="len")
        .fill_null(0)
        .sort("stratum")
    )
    cols = ["stratum"] + [c for c in ("train", "val", "test") if c in balance.columns]
    logger.info("Split balance by stratum:\n" + str(balance.select(cols)))

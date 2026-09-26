"""Create the patient-level splits used in the paper.

    uv run python -m false_confidence.splits.create [--out-dir DIR]

Two split files are produced:

``split_mix`` (M_mix, N = 1,142)
    1. Stratify *patients* by race (White/Black) x age (<40/40-60/60+) x sex and
       assign 70/10/20 train/val/test within each stratum. Patients of any
       other race share one stratum. All exams of a patient stay in one split.
    2. Sex-balance every split by randomly dropping female exams until the
       female and male exam counts are equal (1,254 -> 1,142 exams).
    3. Tag each exam with its label provenance (gold/silver) from MIDRC.

``split_gold`` (M_gold, N = 408)
    The gold exams of ``split_mix``, sex-balanced again within each split.

.. note::
   The split files shipped in ``splits/`` are the exact ones used in the paper
   and should be used to reproduce it. The original implementation iterated
   over an unordered ``unique()``, so re-running it did not return the same
   patient order; this version fixes the order (``maintain_order=True``) and is
   deterministic, but it does not recreate the historical random draw. By
   default this script therefore writes to ``outputs/splits`` rather than
   overwriting ``splits/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from false_confidence.data.loader import load_metadata
from false_confidence.data.schemas import Col
from false_confidence.logger import get_logger
from false_confidence.settings import settings
from false_confidence.splits.io import (
    GOLD,
    MIX,
    add_strata,
    log_balance,
    save_splits,
)
from false_confidence.splits.provenance import add_annotation_quality

logger = get_logger(__name__)

VAL_RATIO = 0.10
TEST_RATIO = 0.20


def _assign(patient_ids: list[str], rng: np.random.Generator) -> list[dict]:
    """Shuffle patients and assign the first 20% to test, next 10% to val."""
    n = len(patient_ids)
    shuffled = rng.permutation(patient_ids).tolist()
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    labels = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
    return [{Col.PATIENT_ID: p, "split": s} for p, s in zip(shuffled, labels)]


def stratified_patient_split(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Patient-level 70/10/20 split stratified by race x age x sex."""
    rng = np.random.default_rng(seed)
    exams = add_strata(df)
    patients = exams.select(Col.PATIENT_ID, "stratum").unique(
        subset=[Col.PATIENT_ID], keep="first", maintain_order=True
    )

    assignments: list[dict] = []
    for stratum in sorted(patients["stratum"].unique().to_list()):
        if stratum == "Other":
            continue
        ids = patients.filter(pl.col("stratum") == stratum)[Col.PATIENT_ID].to_list()
        assignments += _assign(ids, rng)
    other = patients.filter(pl.col("stratum") == "Other")[Col.PATIENT_ID].to_list()
    assignments += _assign(other, rng)

    return exams.join(pl.DataFrame(assignments), on=Col.PATIENT_ID, how="left")


def _drop_to_balance(
    df: pl.DataFrame, split: str, rng: np.random.Generator, *, majority: str | None
) -> set[str]:
    """Exam IDs to drop so ``split`` has equal male and female counts.

    ``majority`` fixes which sex is downsampled (``None``: whichever is larger).
    """
    part = df.filter(pl.col("split") == split)
    ids = {
        sex: part.filter(pl.col("sex_bin") == sex)[Col.SERIES_SUBMITTER_ID].to_list()
        for sex in ("Male", "Female")
    }
    if majority is None:
        majority = "Male" if len(ids["Male"]) > len(ids["Female"]) else "Female"
    minority = "Female" if majority == "Male" else "Male"
    n_drop = len(ids[majority]) - len(ids[minority])
    if n_drop <= 0:
        return set()
    logger.info(f"Dropping {n_drop} {majority} exams from {split}")
    return set(rng.choice(ids[majority], size=n_drop, replace=False).tolist())


def create_split_mix(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Stratified patient split, sex-balanced by dropping female exams."""
    result = stratified_patient_split(df, seed)
    rng = np.random.default_rng(seed)
    drop: set[str] = set()
    for split in ("train", "val", "test"):
        drop |= _drop_to_balance(result, split, rng, majority="Female")
    result = result.filter(~pl.col(Col.SERIES_SUBMITTER_ID).is_in(drop))
    return add_annotation_quality(result)


def create_split_gold(split_mix: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Gold exams of ``split_mix``, re-balanced by sex within each split."""
    gold = split_mix.filter(pl.col("annotation_quality") == "gold")
    rng = np.random.default_rng(seed)
    drop: set[str] = set()
    for split in ("train", "val", "test"):
        drop |= _drop_to_balance(gold, split, rng, majority=None)
    return gold.filter(~pl.col(Col.SERIES_SUBMITTER_ID).is_in(drop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=settings.OUTPUT_DIR / "splits")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    args = parser.parse_args()

    mix = create_split_mix(load_metadata(), args.seed)
    log_balance(mix)
    gold = create_split_gold(mix, args.seed)
    log_balance(gold)

    save_splits(mix, MIX, args.out_dir)
    save_splits(gold, GOLD, args.out_dir)


if __name__ == "__main__":
    main()

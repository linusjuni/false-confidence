"""Score NIfTI predictions against a reference label set (a "ruler").

Computes per-case Dice, HD95 (mm) and nDSC (Raina et al., 2023) for the
vertebral body (label 1) and intervertebral disc (label 2), and writes one CSV
row per case. nDSC's effective class load is estimated from all cases in the
run, so it depends on the evaluated set.

Usage:
    uv run python -m false_confidence.fairness.evaluate \
        --predictions <dir> --references <dir> \
        --mapping $nnUNet_raw/Dataset001_CSpineSeg/case_id_mapping.json \
        --output outputs/eval.csv --metrics dice hd95 ndsc --workers 8
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

from false_confidence.fairness import LABELS
from false_confidence.logger import get_logger

logger = get_logger("fairness.evaluate")

VALID_METRICS = {"dice", "hd95", "ndsc"}


def dice_coefficient(pred: np.ndarray, ref: np.ndarray, label: int) -> float:
    """Binary Dice coefficient for a single label.

    Returns NaN if both masks are empty (nnU-Net convention).
    Returns 0.0 if one mask is empty and the other is not.
    """
    pred_mask = pred == label
    ref_mask = ref == label

    pred_sum = pred_mask.sum()
    ref_sum = ref_mask.sum()

    if pred_sum == 0 and ref_sum == 0:
        return float("nan")
    if pred_sum == 0 or ref_sum == 0:
        return 0.0

    intersection = (pred_mask & ref_mask).sum()
    return float(2.0 * intersection / (pred_sum + ref_sum))


def hausdorff_95(
    pred: np.ndarray,
    ref: np.ndarray,
    label: int,
    spacing: tuple[float, ...],
) -> float:
    """95th-percentile Hausdorff distance in mm for a single label.

    Returns NaN if both masks are empty.
    Returns inf if exactly one mask is empty.
    """
    from surface_distance import (
        compute_robust_hausdorff,
        compute_surface_distances,
    )

    pred_mask = pred == label
    ref_mask = ref == label

    pred_any = pred_mask.any()
    ref_any = ref_mask.any()

    if not pred_any and not ref_any:
        return float("nan")
    if not pred_any or not ref_any:
        return float("inf")

    distances = compute_surface_distances(ref_mask, pred_mask, spacing_mm=spacing)
    return float(compute_robust_hausdorff(distances, 95.0))


def _segmentation_counts(pred: np.ndarray, ref: np.ndarray, label: int) -> dict[str, int]:
    """TP, FP, FN counts for a single label (used by nDSC)."""
    pred_mask = pred == label
    ref_mask = ref == label
    tp = int((pred_mask & ref_mask).sum())
    fp = int((pred_mask & ~ref_mask).sum())
    fn = int((~pred_mask & ref_mask).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "ref_vol": int(ref_mask.sum())}


def _compute_ndsc(df: pl.DataFrame) -> pl.DataFrame:
    """Compute nDSC from intermediate counts (Raina et al. 2023).

    Two-pass: first derives effective_load per label from the dataset,
    then computes nDSC per case. Drops intermediate columns.
    """
    for label_name in LABELS.values():
        tp = f"_tp_{label_name}"
        fp = f"_fp_{label_name}"
        fn = f"_fn_{label_name}"
        rv = f"_ref_vol_{label_name}"
        tv = "_total_voxels"

        effective_load = float(df.select((pl.col(rv).cast(pl.Float64) / pl.col(tv)).mean()).item())
        logger.info("nDSC effective_load", label=label_name, r=f"{effective_load:.6f}")

        non_ref = pl.col(tv).cast(pl.Float64) - pl.col(rv)
        kappa = (1.0 - effective_load) * pl.col(rv) / (effective_load * non_ref)

        ndsc_expr = 2.0 * pl.col(tp) / (kappa * pl.col(fp) + 2.0 * pl.col(tp) + pl.col(fn))

        ndsc_expr = (
            pl.when((pl.col(tp) == 0) & (pl.col(rv) == 0))
            .then(float("nan"))
            .when((pl.col(tp) == 0) | (pl.col(rv) == 0))
            .then(0.0)
            .otherwise(ndsc_expr)
        )

        df = df.with_columns(ndsc_expr.alias(f"ndsc_{label_name}"))

    df = df.drop([c for c in df.columns if c.startswith("_")])
    return df


def evaluate_case(
    pred_path: Path,
    ref_path: Path,
    case_id: str,
    series_submitter_id: str,
    metrics: set[str] | None = None,
) -> dict:
    """Load one prediction/reference NIfTI pair and compute all requested metrics."""
    if metrics is None:
        metrics = {"dice"}

    ref_img = nib.load(ref_path)
    ref_data = ref_img.get_fdata().astype(np.uint8)
    pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

    if ref_data.shape != pred_data.shape:
        msg = f"Shape mismatch for {case_id}: pred {pred_data.shape} vs ref {ref_data.shape}"
        raise ValueError(msg)

    result: dict = {
        "case_id": case_id,
        "series_submitter_id": series_submitter_id,
    }

    if "ndsc" in metrics:
        result["_total_voxels"] = int(ref_data.size)

    for label_int, label_name in LABELS.items():
        if "dice" in metrics:
            result[f"dice_{label_name}"] = dice_coefficient(pred_data, ref_data, label_int)

        if "hd95" in metrics:
            spacing = ref_img.header.get_zooms()[:3]
            result[f"hd95_{label_name}"] = hausdorff_95(
                pred_data, ref_data, label_int, tuple(float(s) for s in spacing)
            )

        if "ndsc" in metrics:
            counts = _segmentation_counts(pred_data, ref_data, label_int)
            result[f"_tp_{label_name}"] = counts["tp"]
            result[f"_fp_{label_name}"] = counts["fp"]
            result[f"_fn_{label_name}"] = counts["fn"]
            result[f"_ref_vol_{label_name}"] = counts["ref_vol"]

    return result


def _evaluate_case_args(args: tuple) -> dict | None:
    """Wrapper for ProcessPoolExecutor (top-level function for pickling)."""
    pred_path, ref_path, case_id, series_submitter_id, metrics = args
    return evaluate_case(
        Path(pred_path),
        Path(ref_path),
        case_id,
        series_submitter_id,
        metrics=metrics,
    )


def evaluate_folder(
    pred_dir: Path,
    ref_dir: Path,
    mapping: list[dict],
    output_path: Path | None = None,
    metrics: set[str] | None = None,
    split: str = "test",
    workers: int = 1,
) -> pl.DataFrame:
    """Evaluate all matching prediction/reference pairs in the directories.

    Only evaluates cases where both prediction and reference files exist
    and the mapping entry matches the requested split.
    """
    if metrics is None:
        metrics = {"dice"}

    invalid = metrics - VALID_METRICS
    if invalid:
        msg = f"Unknown metrics: {invalid}. Valid: {VALID_METRICS}"
        raise ValueError(msg)

    filtered = [e for e in mapping if e["split"] == split]
    logger.info(
        "Evaluating", split=split, cases=len(filtered), metrics=sorted(metrics), workers=workers
    )

    work_items: list[tuple] = []
    for entry in filtered:
        case_id = entry["case_id"]
        pred_path = pred_dir / f"{case_id}.nii.gz"
        ref_path = ref_dir / f"{case_id}.nii.gz"

        if not pred_path.exists():
            logger.warning("Missing prediction", case_id=case_id)
            continue
        if not ref_path.exists():
            logger.warning("Missing reference", case_id=case_id)
            continue

        work_items.append(
            (
                str(pred_path),
                str(ref_path),
                case_id,
                entry["series_submitter_id"],
                metrics,
            )
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_evaluate_case_args, work_items))
    else:
        results = [_evaluate_case_args(item) for item in work_items]

    df = pl.DataFrame(results)

    if "ndsc" in metrics:
        df = _compute_ndsc(df)

    logger.success("Evaluation complete", cases=df.height)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_path)
        logger.info("Saved CSV", path=str(output_path))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate NIfTI predictions against reference labels"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True, help="case_id_mapping.json")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["dice"],
        choices=sorted(VALID_METRICS),
        help="Metrics to compute (default: dice only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for evaluation (default: 1)",
    )
    args = parser.parse_args()

    mapping_data = json.loads(args.mapping.read_text())
    evaluate_folder(
        pred_dir=args.predictions,
        ref_dir=args.references,
        mapping=mapping_data,
        output_path=args.output,
        metrics=set(args.metrics),
        split=args.split,
        workers=args.workers,
    )

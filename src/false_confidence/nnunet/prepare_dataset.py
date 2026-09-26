"""
Build an nnU-Net raw dataset directory from the CSpineSeg download.

Reads the dataset's split file (``splits/split_mix.tsv`` or ``split_gold.tsv``)
and, under ``$nnUNet_raw/<DatasetNNN>/``:
  1. links train+val images and masks into imagesTr/ and labelsTr/
     (nnU-Net cross-validates over the pooled development set),
  2. links test images into imagesTs/ and their reference masks into labelsTs/,
  3. writes dataset.json and case_id_mapping.json (nnU-Net case ID <-> series).

For Dataset002 (gold only), labelsTs/ holds the 76 expert-labelled test cases:
the gold ruler of experiment E2.

Usage:
    uv run python -m false_confidence.nnunet.prepare_dataset --dataset-id 1 [--copy]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

from false_confidence.data.loader import load_annotation_filenames
from false_confidence.data.schemas import Col
from false_confidence.logger import get_logger
from false_confidence.nnunet import DATASETS
from false_confidence.settings import settings
from false_confidence.splits import load_splits

logger = get_logger(__name__)

# nnU-Net naming: cspine_NNNNNN where NNNNNN is extracted from the source filename
_PATIENT_NUM_RE = re.compile(r"593973-(\d{6})")


def _source_to_case_id(filename: str) -> str:
    """Convert a source filename to an nnU-Net case identifier.

    '593973-000123_Study-MR-1_Series-22.nii.gz' -> 'cspine_000123'

    For multi-exam patients (same patient number, different study), the study
    and series numbers are hashed into an extra suffix to keep identifiers unique.
    """
    m = _PATIENT_NUM_RE.search(filename)
    if not m:
        msg = f"Cannot extract patient number from filename: {filename}"
        raise ValueError(msg)
    return f"cspine_{m.group(1)}"


def _ensure_unique_ids(
    filenames: list[str],
) -> dict[str, str]:
    """Map source filenames to unique nnU-Net case identifiers.

    Most patients have one exam so their case_id is simply cspine_NNNNNN.
    For the 23 multi-exam patients, we append _s{N} to disambiguate.
    """
    raw_mapping: dict[str, str] = {}
    for fn in filenames:
        raw_mapping[fn] = _source_to_case_id(fn)

    # Detect collisions and disambiguate
    id_to_filenames: dict[str, list[str]] = {}
    for fn, cid in raw_mapping.items():
        id_to_filenames.setdefault(cid, []).append(fn)

    mapping: dict[str, str] = {}
    for cid, fns in id_to_filenames.items():
        if len(fns) == 1:
            mapping[fns[0]] = cid
        else:
            # Sort for determinism, then append _s0, _s1, ...
            for i, fn in enumerate(sorted(fns)):
                mapping[fn] = f"{cid}_s{i}"

    return mapping


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    """Create a symlink or copy a file."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare_dataset(*, dataset_id: int = 1, copy: bool = False) -> Path:
    """Build an nnU-Net dataset directory under $nnUNet_raw.

    Args:
        dataset_id: Dataset ID (1 = M_mix, 2 = M_gold).
        copy: If True, copy files instead of symlinking.

    Returns:
        Path to the created dataset directory.
    """
    config = DATASETS[dataset_id]
    dataset_name = config["name"]
    split_version = config["split"]

    dataset_dir = settings.nnUNet_raw / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    # Load split assignments
    splits = load_splits(split_version, with_strata=False)
    logger.info("Loaded splits", version=split_version, rows=splits.height)

    train_val = splits.filter(pl.col("split").is_in(["train", "val"]))
    test = splits.filter(pl.col("split") == "test")

    # Join NIfTI filenames onto the split
    annotations = load_annotation_filenames()
    train_val_df = train_val.join(annotations, on=Col.SERIES_SUBMITTER_ID, how="left")
    test_df = test.join(annotations, on=Col.SERIES_SUBMITTER_ID, how="left")

    all_filenames = train_val_df[Col.FILENAME].to_list() + test_df[Col.FILENAME].to_list()
    case_id_map = _ensure_unique_ids(all_filenames)

    # Symlink/copy training + validation images and labels
    n_tr = 0
    for row in train_val_df.iter_rows(named=True):
        fn = row[Col.FILENAME]
        cid = case_id_map[fn]

        # Image: cspine_NNNNNN_0000.nii.gz
        src_img = settings.annotation_dir / fn
        dst_img = images_tr / f"{cid}_0000.nii.gz"
        _link_or_copy(src_img, dst_img, copy=copy)

        # Label: cspine_NNNNNN.nii.gz (no channel suffix)
        seg_fn = fn.replace(".nii.gz", "_SEG.nii.gz")
        src_seg = settings.segmentation_dir / seg_fn
        dst_seg = labels_tr / f"{cid}.nii.gz"
        _link_or_copy(src_seg, dst_seg, copy=copy)

        n_tr += 1

    logger.success("Linked training+val images and labels", count=n_tr)

    # Symlink/copy test images, and their reference masks into labelsTs/ (used
    # only for evaluation; nnU-Net never reads labelsTs/)
    n_ts = 0
    for row in test_df.iter_rows(named=True):
        fn = row[Col.FILENAME]
        cid = case_id_map[fn]

        src_img = settings.annotation_dir / fn
        dst_img = images_ts / f"{cid}_0000.nii.gz"
        _link_or_copy(src_img, dst_img, copy=copy)

        src_seg = settings.segmentation_dir / fn.replace(".nii.gz", "_SEG.nii.gz")
        _link_or_copy(src_seg, labels_ts / f"{cid}.nii.gz", copy=copy)

        n_ts += 1

    logger.success("Linked test images and reference labels", count=n_ts)

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {
            "background": 0,
            "vertebral_body": 1,
            "disc": 2,
        },
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
    }
    dataset_json_path = dataset_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=4))
    logger.success("Wrote dataset.json", path=str(dataset_json_path))

    # Write case_id_mapping.json (source filename <-> nnU-Net case identifier)
    # Include split assignment for traceability
    mapping_records = []
    for row in splits.iter_rows(named=True):
        sid = row[Col.SERIES_SUBMITTER_ID]
        fn_rows = annotations.filter(pl.col(Col.SERIES_SUBMITTER_ID) == sid)
        if fn_rows.height == 0:
            continue
        fn = fn_rows[Col.FILENAME][0]
        cid = case_id_map.get(fn)
        if cid is None:
            continue
        mapping_records.append(
            {
                "case_id": cid,
                "source_filename": fn,
                "series_submitter_id": sid,
                "patient_id": row[Col.PATIENT_ID],
                "split": row["split"],
            }
        )

    mapping_path = dataset_dir / "case_id_mapping.json"
    mapping_path.write_text(json.dumps(mapping_records, indent=2))
    logger.success(
        "Wrote case_id_mapping.json", entries=len(mapping_records), path=str(mapping_path)
    )

    # Summary
    logger.success(
        "Dataset prepared",
        dataset=dataset_name,
        train_val=n_tr,
        test=n_ts,
        mode="copy" if copy else "symlink",
        path=str(dataset_dir),
    )

    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare nnU-Net dataset from CSpineSeg")
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        choices=list(DATASETS),
        help="1 = M_mix (gold + silver), 2 = M_gold (gold only)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (uses ~5 GB extra disk space)",
    )
    args = parser.parse_args()
    prepare_dataset(dataset_id=args.dataset_id, copy=args.copy)

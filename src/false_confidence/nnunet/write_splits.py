"""
Create a demographically stratified splits_final.json for nnU-Net 5-fold CV.

Reads the split TSV and case_id_mapping.json, then creates 5 folds where each
fold preserves the race x age x sex strata proportions. The output is written
to $nnUNet_preprocessed/<DatasetNNN>/splits_final.json.

Must be run AFTER nnUNetv2_plan_and_preprocess (which creates the preprocessed
directory) and BEFORE nnUNetv2_train.

Usage:
    uv run python -m false_confidence.nnunet.write_splits --dataset-id 1
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from false_confidence.data.schemas import Col
from false_confidence.logger import get_logger
from false_confidence.nnunet import DATASETS
from false_confidence.settings import settings
from false_confidence.splits import load_splits

logger = get_logger(__name__)

N_FOLDS = 5


def create_nnunet_splits(dataset_id: int = 1) -> list[dict[str, list[str]]]:
    """Create 5-fold stratified splits for nnU-Net.

    Only train+val cases go into imagesTr/ and thus into the nnU-Net folds.
    Test cases are in imagesTs/ and never appear here.

    Args:
        dataset_id: Dataset ID (1 = M_mix, 2 = M_gold).

    Returns:
        List of 5 dicts, each with 'train' and 'val' keys mapping to lists
        of nnU-Net case identifier strings.
    """
    config = DATASETS[dataset_id]
    dataset_name = config["name"]
    split_version = config["split"]

    dataset_dir = settings.nnUNet_raw / dataset_name

    # Load case_id_mapping.json (written by prepare_dataset)
    mapping_path = dataset_dir / "case_id_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"case_id_mapping.json not found at {mapping_path}. "
            "Run `python -m false_confidence.nnunet.prepare_dataset` first."
        )
    mapping = json.loads(mapping_path.read_text())

    splits = load_splits(split_version)

    # Filter to train+val only (what's in imagesTr/)
    train_val_sids = {r["series_submitter_id"] for r in mapping if r["split"] in ("train", "val")}

    train_val = splits.filter(pl.col(Col.SERIES_SUBMITTER_ID).is_in(train_val_sids))

    # Build a lookup from series_submitter_id -> case_id
    sid_to_cid = {r["series_submitter_id"]: r["case_id"] for r in mapping}

    # Build stratum labels for stratified k-fold
    case_ids = []
    strata = []
    for row in train_val.iter_rows(named=True):
        sid = row[Col.SERIES_SUBMITTER_ID]
        cid = sid_to_cid.get(sid)
        if cid is None:
            continue
        case_ids.append(cid)
        strata.append(row["stratum"])

    case_ids = np.array(case_ids)
    strata = np.array(strata)

    logger.info(
        "Stratifying",
        n_cases=len(case_ids),
        n_strata=len(np.unique(strata)),
        strata=sorted(np.unique(strata).tolist()),
    )

    # Stratified k-fold matching nnU-Net's default random_state
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED)

    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(case_ids, strata)):
        train_ids = sorted(case_ids[train_indices].tolist())
        val_ids = sorted(case_ids[val_indices].tolist())
        folds.append({"train": train_ids, "val": val_ids})
        logger.info(
            f"Fold {fold_idx}",
            train=len(train_ids),
            val=len(val_ids),
        )

    return folds


def write_splits(dataset_id: int = 1) -> None:
    """Create and write splits_final.json to the nnU-Net preprocessed directory."""
    config = DATASETS[dataset_id]
    dataset_name = config["name"]
    preprocessed_dir = settings.nnUNet_preprocessed / dataset_name

    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed directory not found: {preprocessed_dir}. "
            f"Run `nnUNetv2_plan_and_preprocess -d {dataset_id} -pl nnUNetPlannerResEncL` first."
        )

    folds = create_nnunet_splits(dataset_id)

    splits_path = preprocessed_dir / "splits_final.json"
    splits_path.write_text(json.dumps(folds, indent=2))
    logger.success("Wrote splits_final.json", path=str(splits_path), n_folds=len(folds))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write stratified nnU-Net CV splits")
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        choices=list(DATASETS),
        help="1 = M_mix (gold + silver), 2 = M_gold (gold only)",
    )
    args = parser.parse_args()
    write_splits(args.dataset_id)

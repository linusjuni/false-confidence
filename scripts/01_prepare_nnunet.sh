#!/usr/bin/env bash
# Build the nnU-Net dataset, plan + preprocess with the ResEnc-L planner, and
# write the demographically stratified 5-fold CV splits.
#
# Usage: bash scripts/01_prepare_nnunet.sh DATASET_ID   (1 = M_mix, 2 = M_gold)
set -euo pipefail

DATASET_ID=${1:?Usage: 01_prepare_nnunet.sh DATASET_ID (1 = M_mix, 2 = M_gold)}
set -a; [[ -f .env ]] && source .env; set +a

uv run python -m false_confidence.nnunet.prepare_dataset --dataset-id "${DATASET_ID}"
uv run nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -pl nnUNetPlannerResEncL \
    -c 2d 3d_fullres --verify_dataset_integrity
# Replace nnU-Net's random folds with folds stratified by race x age x sex.
uv run python -m false_confidence.nnunet.write_splits --dataset-id "${DATASET_ID}"

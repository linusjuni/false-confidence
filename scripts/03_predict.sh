#!/usr/bin/env bash
# Predict the held-out test set with the 2D + 3D full-resolution ensemble
# (the configuration nnU-Net selected for both datasets) and apply the
# selected post-processing.
#
# Output: $nnUNet_results/<Dataset>/predictions_test_pp/
# Usage:  bash scripts/03_predict.sh DATASET_ID
set -euo pipefail

DATASET_ID=${1:?Usage: 03_predict.sh DATASET_ID (1 = M_mix, 2 = M_gold)}
set -a; [[ -f .env ]] && source .env; set +a

PLANS=nnUNetResEncUNetLPlans
TRAINER=nnUNetTrainer
NAME=$(uv run python -c "from false_confidence.nnunet import DATASETS; print(DATASETS[${DATASET_ID}]['name'])")
RES="${nnUNet_results}/${NAME}"

uv run nnUNetv2_find_best_configuration "${DATASET_ID}" -c 2d 3d_fullres -p "${PLANS}" -tr "${TRAINER}"

for CONFIG in 2d 3d_fullres; do
    uv run nnUNetv2_predict -d "${NAME}" -i "${nnUNet_raw}/${NAME}/imagesTs" \
        -o "${RES}/predictions_test_${CONFIG}" -f 0 1 2 3 4 \
        -tr "${TRAINER}" -c "${CONFIG}" -p "${PLANS}" --save_probabilities
done

uv run nnUNetv2_ensemble -i "${RES}/predictions_test_2d" "${RES}/predictions_test_3d_fullres" \
    -o "${RES}/predictions_test_ensemble" -np 8

ENS="${RES}/ensembles/ensemble___${TRAINER}__${PLANS}__2d___${TRAINER}__${PLANS}__3d_fullres___0_1_2_3_4"
uv run nnUNetv2_apply_postprocessing -i "${RES}/predictions_test_ensemble" \
    -o "${RES}/predictions_test_pp" -pp_pkl_file "${ENS}/postprocessing.pkl" \
    -plans_json "${ENS}/plans.json" -np 8

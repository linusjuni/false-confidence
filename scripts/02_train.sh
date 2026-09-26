#!/usr/bin/env bash
# Train one nnU-Net fold. The paper trains 2 configs x 5 folds per dataset:
#
#   for c in 2d 3d_fullres; do for f in 0 1 2 3 4; do
#       bash scripts/02_train.sh 1 $c $f
#   done; done
#
# Usage: bash scripts/02_train.sh DATASET_ID CONFIG FOLD
set -euo pipefail

DATASET_ID=${1:?Usage: 02_train.sh DATASET_ID CONFIG FOLD}
CONFIG=${2:?Usage: 02_train.sh DATASET_ID CONFIG FOLD}
FOLD=${3:?Usage: 02_train.sh DATASET_ID CONFIG FOLD}
set -a; [[ -f .env ]] && source .env; set +a

# Fewer augmentation workers avoid a known batchgenerators crash with large
# 3D patches (MIC-DKFZ/nnUNet#2523).
export OMP_NUM_THREADS=1
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-2}

uv run nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz -p nnUNetResEncUNetLPlans

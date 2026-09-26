#!/usr/bin/env bash
# Experiments E1 and E2.
#
#   E1  global audit: M_mix on the full 228-case test set, dataset's own labels
#   E2  biased ruler: M_mix's predictions on the 76 gold-test images, scored
#       against expert labels (gold ruler) and against M_gold's predictions
#       (silver ruler)
#
# Test cases are enumerated from the dataset that owns the reference labels
# (Dataset002 for E2); both datasets share case IDs. Per-case metric CSVs are written to $EVAL_DIR (default: outputs/eval) and
# reused if present, so the analysis can be re-run without images or models.
#
# Usage: bash scripts/04_audit.sh
set -euo pipefail
set -a; [[ -f .env ]] && source .env; set +a

EVAL_DIR=${EVAL_DIR:-outputs/eval}
WORKERS=${WORKERS:-8}
D1="Dataset001_CSpineSeg"
D2="Dataset002_CSpineSeg_Gold"

evaluate () {  # evaluate OUT_CSV PREDICTIONS REFERENCES DATASET_FOR_CASE_MAPPING
    if [[ -f "$1" ]]; then
        echo "Reusing $1"
        return
    fi
    uv run python -m false_confidence.fairness.evaluate --output "$1" \
        --predictions "$2" --references "$3" --mapping "${nnUNet_raw}/$4/case_id_mapping.json" \
        --metrics dice hd95 ndsc --workers "${WORKERS}"
}

evaluate "${EVAL_DIR}/eval_global.csv" \
    "${nnUNet_results}/${D1}/predictions_test_pp" "${nnUNet_raw}/${D1}/labelsTs" "${D1}"
evaluate "${EVAL_DIR}/eval_ruler_gold.csv" \
    "${nnUNet_results}/${D1}/predictions_test_pp" "${nnUNet_raw}/${D2}/labelsTs" "${D2}"
evaluate "${EVAL_DIR}/eval_ruler_silver.csv" \
    "${nnUNet_results}/${D1}/predictions_test_pp" "${nnUNet_results}/${D2}/predictions_test_pp" "${D2}"

# E1
uv run python -m false_confidence.fairness.analyze --report-name fairness_global \
    --evaluation-csvs "${EVAL_DIR}/eval_global.csv" --ruler-labels global

# E2 (the labels must be exactly `gold` and `silver`)
uv run python -m false_confidence.fairness.analyze --report-name fairness_biased_ruler \
    --evaluation-csvs "${EVAL_DIR}/eval_ruler_gold.csv" "${EVAL_DIR}/eval_ruler_silver.csv" \
    --ruler-labels gold silver

uv run python scripts/plot_age_trend.py

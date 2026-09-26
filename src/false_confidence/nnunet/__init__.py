"""nnU-Net v2 integration: dataset folders, cross-validation folds, test labels."""

from false_confidence.splits import GOLD, MIX

# The two training regimes of the paper. Both use the same architecture,
# recipe and stratification and differ only in the labels they see.
DATASETS = {
    1: {"name": "Dataset001_CSpineSeg", "split": MIX},  # M_mix: gold + silver
    2: {"name": "Dataset002_CSpineSeg_Gold", "split": GOLD},  # M_gold: gold only
}

PLANS = "nnUNetResEncUNetLPlans"
CONFIGS = ("2d", "3d_fullres")

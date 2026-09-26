"""Patient-level train/val/test splits and gold/silver label provenance."""

from false_confidence.splits.io import (
    GOLD,
    MIX,
    add_strata,
    apply_splits,
    load_splits,
    save_splits,
)

__all__ = ["MIX", "GOLD", "add_strata", "apply_splits", "load_splits", "save_splits"]

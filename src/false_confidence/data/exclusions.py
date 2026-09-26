import polars as pl

# Series IDs to exclude from all analyses
EXCLUDED_SERIES_IDS = [
    # Excluded 2026-02-10: Localizer/scout scan with abnormal dimensions
    # - Dimensions: 56×512×512 voxels (vs typical 512×512×14 for cervical spine)
    # - Physical size: 56mm×250mm×250mm (FOV too narrow, depth too large)
    # - Issue: Either wrong scan type (localizer instead of diagnostic) or
    #   misoriented axes in NIfTI header
    # - Impact: Fundamentally different from rest of dataset (extreme outlier)
    # - Decision: Exclude to maintain data quality
    "1.2.826.0.1.3680043.10.474.593973.22529",
]


def filter_excluded_cases(df, logger=None):
    """
    Filter out excluded series IDs from a DataFrame.

    Args:
        df: Polars DataFrame with 'series_submitter_id' column
        logger: Optional logger for logging exclusion info

    Returns:
        Filtered DataFrame with excluded cases removed
    """

    n_before = df.height
    df = df.filter(~pl.col("series_submitter_id").is_in(EXCLUDED_SERIES_IDS))
    n_excluded = n_before - df.height

    if n_excluded > 0 and logger is not None:
        logger.info(f"Excluded {n_excluded} outlier cases", total=df.height)

    return df

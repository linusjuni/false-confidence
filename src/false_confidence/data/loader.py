"""
Join logic:
All three TSVs have a "submitter_id" column, but it means something different
in each:
    - patient ID in case_RSNA_20250321.tsv
    - study UID in imaging_study_RSNA_20250321.tsv
    - series UID in mr_series_RSNA_20250321.tsv

The foreign keys that link them are:
    - series["imaging_studies.submitter_id"]  ==  study["submitter_id"]   (study UIDs)
    - study["case_ids"]                       ==  case["submitter_id"]    (patient IDs)
We rename these to a shared name (study_submitter_id / patient_id) so polars
can join on a single column. Columns that collide (type, case_ids,
image_data_modified) are prefixed per table. Redundant copies of join keys
are dropped after merging.

The annotation_file TSV is joined to add NIfTI filenames (needed for nnU-Net
prep and evaluation). It maps series_submitter_id → filename.
"""

from pathlib import Path

import polars as pl

from false_confidence.data.exclusions import filter_excluded_cases
from false_confidence.data.schemas import Col, ExamSchema
from false_confidence.logger import get_logger
from false_confidence.settings import settings

logger = get_logger(__name__)


def structured_tsv(table: str) -> Path:
    """Locate a MIDRC metadata TSV by table name, e.g. ``case`` -> ``case_RSNA_*.tsv``.

    The MIDRC export stamps each file with a release date, so we match on the
    table prefix rather than a fixed filename.
    """
    matches = sorted(settings.structured_dir.glob(f"{table}_*.tsv"))
    if not matches:
        raise FileNotFoundError(
            f"No '{table}_*.tsv' in {settings.structured_dir}. "
            "Download the CSpineSeg metadata and set DATA_DIR (see README)."
        )
    return matches[-1]


def load_annotation_filenames() -> pl.DataFrame:
    """Load annotation_file TSV and return filename → series mapping.

    Filters to image files only (excludes _SEG segmentation masks).
    """
    df = pl.read_csv(structured_tsv("annotation_file"), separator="\t")
    df = df.filter(~pl.col("file_name").str.ends_with("_SEG.nii.gz")).select(
        pl.col("file_name").alias(Col.FILENAME),
        pl.col("mr_series_files.submitter_id").alias(Col.SERIES_SUBMITTER_ID),
    )
    logger.success("Loaded annotation filenames", rows=df.height)
    return df


def load_metadata() -> pl.DataFrame:
    """Load and merge metadata TSVs into one exam-level DataFrame.

    Joins series, study, case, and annotation-file TSVs. Applies
    exclusions and validates the result against ExamSchema.

    Returns:
        Validated Polars DataFrame with one row per exam (~1,254 rows,
        ~71 columns). Key columns are type-checked; extra MIDRC platform
        columns pass through untouched.
    """
    cases = pl.read_csv(structured_tsv("case"), separator="\t").rename(
        {"submitter_id": "patient_id", "type": "case_type"}
    )

    studies = pl.read_csv(structured_tsv("imaging_study"), separator="\t").rename(
        {
            "submitter_id": "study_submitter_id",
            "case_ids": "patient_id",
            "type": "study_type",
        }
    )

    series = pl.read_csv(structured_tsv("mr_series"), separator="\t").rename(
        {
            "imaging_studies.submitter_id": "study_submitter_id",
            "type": "series_type",
            "submitter_id": "series_submitter_id",
            "case_ids": "patient_id",
            "image_data_modified": "series_image_data_modified",
        }
    )

    annotations = load_annotation_filenames()

    df = (
        series.join(studies, on="study_submitter_id", how="left")
        .join(cases, on="patient_id", how="left")
        .join(annotations, on="series_submitter_id", how="left")
        .drop("study_submitter_id", "patient_id_right", "cases.submitter_id", "case_ids")
        # Normalize manufacturer casing ("Siemens" → "SIEMENS")
        .with_columns(pl.col("manufacturer").str.to_uppercase())
    )

    df = filter_excluded_cases(df, logger)

    # Validate against schema (extra MIDRC columns pass through)
    df = ExamSchema.validate(df, allow_superfluous_columns=True)

    logger.success("Loaded metadata", rows=df.height, cols=df.width)
    return df

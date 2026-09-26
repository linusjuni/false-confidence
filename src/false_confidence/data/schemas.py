"""
Data schema definitions for the CSpineSeg dataset.

Provides:
    - Col: StrEnum of column names referenced in analysis code.
    - Race, Sex, Ethnicity, Manufacturer: StrEnums of valid categorical values.
    - ExamSchema: Patito model that validates the merged exam-level DataFrame.
    - SegmentationVolumeSchema / SegmentationVolumeCol: per-exam mask volumes.

The ExamSchema is validated once at load time (in loader.py). Downstream code
uses Col and value enums for type-safe, string-free DataFrame access.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import patito as pt

# ---------------------------------------------------------------------------
# Column name enum (only columns we reference in code)
# ---------------------------------------------------------------------------


class Col(StrEnum):
    """Column names in the merged exam-level DataFrame."""

    # Identifiers
    PATIENT_ID = "patient_id"
    SERIES_SUBMITTER_ID = "series_submitter_id"
    SERIES_UID = "series_uid"
    FILENAME = "filename"

    # Demographics
    RACE = "race"
    SEX = "sex"
    ETHNICITY = "ethnicity"
    AGE = "age_at_imaging"
    AGE_GT89 = "age_at_imaging_gt89"

    # Scanner
    MANUFACTURER = "manufacturer"
    MANUFACTURER_MODEL = "manufacturer_model_name"
    FIELD_STRENGTH = "magnetic_field_strength"
    SLICE_THICKNESS = "slice_thickness"
    PIXEL_SPACING = "pixel_spacing"
    SPACING_BETWEEN_SLICES = "spacing_between_slices"
    ECHO_TIME = "echo_time"
    REPETITION_TIME = "repetition_time"


class SegmentationVolumeSchema(pt.Model):
    """Schema for the segmentation volume properties DataFrame."""

    filename: str = pt.Field(unique=True)
    series_submitter_id: str = pt.Field(unique=True)
    n_voxels_vertebral_body: int
    n_voxels_disc: int
    volume_mm3_vertebral_body: float
    volume_mm3_disc: float
    n_components_vertebral_body: int
    n_components_disc: int


class SegmentationVolumeCol(StrEnum):
    """Column names in the segmentation volume properties DataFrame."""

    # Voxel counts
    N_VOXELS_VERTEBRAL_BODY = "n_voxels_vertebral_body"
    N_VOXELS_DISC = "n_voxels_disc"

    # Physical volumes (mm³)
    VOLUME_MM3_VERTEBRAL_BODY = "volume_mm3_vertebral_body"
    VOLUME_MM3_DISC = "volume_mm3_disc"

    # Connected components (individual structures)
    N_COMPONENTS_VERTEBRAL_BODY = "n_components_vertebral_body"
    N_COMPONENTS_DISC = "n_components_disc"


# ---------------------------------------------------------------------------
# Categorical value enums
# ---------------------------------------------------------------------------


class Race(StrEnum):
    """Valid values in the race column."""

    WHITE = "White"
    BLACK = "Black or African American"
    ASIAN = "Asian"
    AMERICAN_INDIAN = "American Indian or Alaska Native"
    PACIFIC_ISLANDER = "Native Hawaiian or other Pacific Islander"
    OTHER = "Other"
    NOT_REPORTED = "Not Reported"


class Sex(StrEnum):
    """Valid values in the sex column."""

    FEMALE = "Female"
    MALE = "Male"


class Ethnicity(StrEnum):
    """Valid values in the ethnicity column."""

    NOT_HISPANIC = "Not Hispanic or Latino"
    HISPANIC = "Hispanic or Latino"
    NOT_REPORTED = "Not Reported"


class Manufacturer(StrEnum):
    """Valid values in the manufacturer column (post-normalization)."""

    SIEMENS = "SIEMENS"
    GE = "GE MEDICAL SYSTEMS"


# ---------------------------------------------------------------------------
# Patito schema — validated at load time
# ---------------------------------------------------------------------------


class ExamSchema(pt.Model):
    """Schema for the merged exam-level DataFrame.

    Only the columns we actively use in analysis are modeled here.
    The full DataFrame has ~70 columns; extra columns pass through
    validation untouched via allow_superfluous_columns=True.

    age_at_imaging is Optional because 13 patients aged >89 have
    their age redacted (null) for privacy, indicated by
    age_at_imaging_gt89 == "Yes".
    """

    # Identifiers
    patient_id: str
    series_submitter_id: str = pt.Field(unique=True)
    series_uid: str
    filename: str = pt.Field(unique=True)

    # Demographics
    race: Literal[
        Race.WHITE,
        Race.BLACK,
        Race.ASIAN,
        Race.AMERICAN_INDIAN,
        Race.PACIFIC_ISLANDER,
        Race.OTHER,
        Race.NOT_REPORTED,
    ]
    sex: Literal[Sex.FEMALE, Sex.MALE]
    ethnicity: Literal[
        Ethnicity.NOT_HISPANIC,
        Ethnicity.HISPANIC,
        Ethnicity.NOT_REPORTED,
    ]
    age_at_imaging: float | None
    age_at_imaging_gt89: str

    # Scanner
    manufacturer: Literal[Manufacturer.SIEMENS, Manufacturer.GE]
    manufacturer_model_name: str
    magnetic_field_strength: float
    slice_thickness: int
    pixel_spacing: float
    spacing_between_slices: float
    echo_time: float
    repetition_time: float

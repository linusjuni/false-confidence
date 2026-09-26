"""Project settings, read from environment variables or a `.env` file.

All paths are relative to the repository root unless overridden. See
`.env.example` for the expected variables.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file_encoding="utf-8", case_sensitive=True)

    RANDOM_SEED: int = Field(default=42)

    # CSpineSeg download (see README, "Data"). Expected layout:
    #   DATA_DIR/annotation/    MRI volumes (*.nii.gz)
    #   DATA_DIR/segmentation/  masks (*_SEG.nii.gz)
    #   DATA_DIR/structured/    MIDRC metadata TSVs
    DATA_DIR: Path = Path("data/cspineseg")

    # Patient-level splits used in the paper (shipped with the repository).
    SPLITS_DIR: Path = Path("splits")

    # Analysis outputs (reports, figures, per-case metric CSVs).
    OUTPUT_DIR: Path = Path("outputs")

    # nnU-Net v2 directories (the same variables nnU-Net itself reads).
    nnUNet_raw: Path = Path("nnunet/raw")
    nnUNet_preprocessed: Path = Path("nnunet/preprocessed")
    nnUNet_results: Path = Path("nnunet/results")

    @property
    def annotation_dir(self) -> Path:
        return self.DATA_DIR / "annotation"

    @property
    def segmentation_dir(self) -> Path:
        return self.DATA_DIR / "segmentation"

    @property
    def structured_dir(self) -> Path:
        return self.DATA_DIR / "structured"

    @property
    def cache_dir(self) -> Path:
        return self.OUTPUT_DIR / "cache"

    @property
    def splits_dir(self) -> Path:
        return self.SPLITS_DIR


settings = Settings()

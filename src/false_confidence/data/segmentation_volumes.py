from __future__ import annotations

import nibabel as nib
import numpy as np
import polars as pl
from scipy.ndimage import label as nd_label

from false_confidence.data.exclusions import filter_excluded_cases
from false_confidence.data.loader import load_annotation_filenames, structured_tsv
from false_confidence.data.schemas import Col, SegmentationVolumeCol, SegmentationVolumeSchema
from false_confidence.logger import get_logger
from false_confidence.settings import settings

logger = get_logger("data.segmentation_volumes")


def extract_segmentation_volume_properties(force_refresh: bool = False) -> pl.DataFrame:
    """
    Extract annotation volume properties from all segmentation masks, with Parquet caching.

    Reads full voxel data from _SEG.nii.gz files for all exams and extracts
    per-label voxel counts, physical volumes (mm³), and connected component counts.

    Labels:
        1 — vertebral bodies
        2 — intervertebral discs

    Cache is invalidated if:
        - force_refresh=True
        - Cache file doesn't exist
        - Cache file is older than annotation_file TSV (staleness check)
    """
    cache_path = settings.cache_dir / "segmentation_volumes.parquet"

    if not force_refresh and cache_path.exists():
        annotation_tsv = structured_tsv("annotation_file")
        if cache_path.stat().st_mtime > annotation_tsv.stat().st_mtime:
            try:
                df = pl.read_parquet(cache_path)
                logger.info("Loaded from cache", rows=df.height)
                return df
            except Exception as e:
                logger.warning("Cache corrupted, recomputing", error=str(e))
                cache_path.unlink(missing_ok=True)

    if not settings.segmentation_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {settings.segmentation_dir}")

    annotation_df = load_annotation_filenames()
    logger.info("Starting segmentation volume extraction", files=annotation_df.height)

    properties = []
    failed_files = []

    for i, row in enumerate(annotation_df.iter_rows(named=True)):
        filename = row[Col.FILENAME]
        series_submitter_id = row[Col.SERIES_SUBMITTER_ID]

        if i % 100 == 0:
            logger.info(f"Processing {i}/{annotation_df.height}")

        seg_filename = filename.replace(".nii.gz", "_SEG.nii.gz")
        path = settings.segmentation_dir / seg_filename

        if not path.exists():
            logger.warning("File not found", filename=seg_filename)
            failed_files.append((seg_filename, "File not found on disk"))
            continue

        try:
            img = nib.load(path)
            data = img.get_fdata().astype(np.uint8)
            zooms = img.header.get_zooms()

            if len(img.shape) != 3:
                logger.warning(
                    "Unexpected shape",
                    filename=seg_filename,
                    shape=img.shape,
                    action="skipping",
                )
                failed_files.append((seg_filename, f"Unexpected shape: {img.shape}"))
                continue

            voxel_vol_mm3 = float(zooms[0]) * float(zooms[1]) * float(zooms[2])

            n_voxels_vb = int((data == 1).sum())
            n_voxels_disc = int((data == 2).sum())

            _, n_comp_vb = nd_label(data == 1)
            _, n_comp_disc = nd_label(data == 2)

            properties.append(
                {
                    Col.FILENAME: filename,
                    Col.SERIES_SUBMITTER_ID: series_submitter_id,
                    SegmentationVolumeCol.N_VOXELS_VERTEBRAL_BODY: n_voxels_vb,
                    SegmentationVolumeCol.N_VOXELS_DISC: n_voxels_disc,
                    SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY: n_voxels_vb * voxel_vol_mm3,
                    SegmentationVolumeCol.VOLUME_MM3_DISC: n_voxels_disc * voxel_vol_mm3,
                    SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY: int(n_comp_vb),
                    SegmentationVolumeCol.N_COMPONENTS_DISC: int(n_comp_disc),
                }
            )

        except Exception as e:
            logger.warning("Failed to load", filename=seg_filename, error=str(e))
            failed_files.append((seg_filename, str(e)))

    if failed_files:
        logger.warning(
            f"Failed to load {len(failed_files)} files",
            success_rate=f"{100 * len(properties) / annotation_df.height:.1f}%",
        )

    df = pl.DataFrame(properties)

    if df.height == 0:
        raise ValueError("No properties extracted - all files failed to load")

    SegmentationVolumeSchema.validate(df)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    logger.success("Cached segmentation volumes", rows=df.height, path=str(cache_path.name))

    return df


def load_segmentation_volumes(force_refresh: bool = False) -> pl.DataFrame:
    """Load segmentation volume properties, extracting if cache is stale or missing."""
    df = extract_segmentation_volume_properties(force_refresh)
    df = filter_excluded_cases(df, logger)
    return df


if __name__ == "__main__":
    df = load_segmentation_volumes()
    print(df.describe())

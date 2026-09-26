"""Anatomical confounders (paper section "Anatomical Confounders").

Measures vertebral-body and disc size from the reference masks and tests how it
varies with sex, race and age, and between scanner manufacturers (voxel counts
vs. physical volume in mm^3). Results land in ``outputs/eda/confounders/``.

Usage (needs the segmentation masks):
    uv run python -m false_confidence.confounders
"""

import numpy as np
import polars as pl
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from false_confidence.data.loader import load_metadata
from false_confidence.data.schemas import Col, SegmentationVolumeCol
from false_confidence.data.segmentation_volumes import load_segmentation_volumes
from false_confidence.report import Report
from false_confidence.stats import kruskal_result, mann_whitney_result

sns.set_theme(style="whitegrid", palette="muted")

AGE_BIN_ORDER = ["<40", "40-60", "60+"]
RACE_ORDER = ["White", "Black or African American"]
SEX_ORDER = ["Female", "Male"]


def run(df, report_name: str) -> None:
    stat_results: list[tuple[str, dict]] = []

    df = df.with_columns(
        pl.when(pl.col(Col.AGE).is_null())
        .then(pl.lit("60+"))
        .when(pl.col(Col.AGE) < 40)
        .then(pl.lit("<40"))
        .when(pl.col(Col.AGE) < 60)
        .then(pl.lit("40-60"))
        .otherwise(pl.lit("60+"))
        .alias("age_bin")
    )

    df_race = df.filter(pl.col(Col.RACE).is_in(RACE_ORDER))

    with Report(report_name) as report:
        # ------------------------------------------------------------------
        # 1. Distributions
        # ------------------------------------------------------------------

        with report.figure("dist_volume_mm3", figsize=(12, 4)) as fig:
            axes = fig.subplots(1, 2)
            sns.histplot(df[SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY], bins=40, ax=axes[0])
            axes[0].set_xlabel("Volume (mm³)")
            axes[0].set_title("Vertebral Body Volume (mm³)")
            sns.histplot(df[SegmentationVolumeCol.VOLUME_MM3_DISC], bins=40, ax=axes[1])
            axes[1].set_xlabel("Volume (mm³)")
            axes[1].set_title("Intervertebral Disc Volume (mm³)")

        with report.figure("dist_voxel_counts", figsize=(12, 4)) as fig:
            axes = fig.subplots(1, 2)
            sns.histplot(df[SegmentationVolumeCol.N_VOXELS_VERTEBRAL_BODY], bins=40, ax=axes[0])
            axes[0].set_xlabel("Voxel count")
            axes[0].set_title("Vertebral Body Voxel Count")
            axes[0].tick_params(axis="x", rotation=45)
            sns.histplot(df[SegmentationVolumeCol.N_VOXELS_DISC], bins=40, ax=axes[1])
            axes[1].set_xlabel("Voxel count")
            axes[1].set_title("Intervertebral Disc Voxel Count")
            axes[1].tick_params(axis="x", rotation=45)

        with report.figure("dist_components", figsize=(12, 4)) as fig:
            axes = fig.subplots(1, 2)
            vb = df[SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY]
            disc = df[SegmentationVolumeCol.N_COMPONENTS_DISC]
            sns.histplot(vb, bins=np.arange(vb.min(), vb.max() + 2) - 0.5, ax=axes[0])
            axes[0].set_xlabel("Component count")
            axes[0].set_title("Vertebral Body Components")
            sns.histplot(disc, bins=np.arange(disc.min(), disc.max() + 2) - 0.5, ax=axes[1])
            axes[1].set_xlabel("Component count")
            axes[1].set_title("Intervertebral Disc Components")

        outliers = df.filter(
            (pl.col(SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY) > 15)
            | (pl.col(SegmentationVolumeCol.N_COMPONENTS_DISC) > 15)
        ).select(
            Col.SERIES_SUBMITTER_ID,
            Col.FILENAME,
            SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY,
            SegmentationVolumeCol.N_COMPONENTS_DISC,
            Col.SEX,
            Col.RACE,
            Col.AGE,
            Col.MANUFACTURER,
            Col.FIELD_STRENGTH,
        )
        report.save_table(outliers, "high_component_outliers")
        report.log_stat("n_high_component_outliers", outliers.height)

        # ------------------------------------------------------------------
        # 2. Volume by demographics (anatomy / confounder check)
        # ------------------------------------------------------------------

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.VOLUME_MM3_DISC),
        ]:
            with report.figure(f"vol_by_sex_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.SEX, col).to_pandas(),
                    x=Col.SEX,
                    y=col,
                    order=SEX_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Sex")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume by Sex")

            a = df.filter(pl.col(Col.SEX) == "Female")[col].drop_nulls()
            b = df.filter(pl.col(Col.SEX) == "Male")[col].drop_nulls()
            res = mann_whitney_result(a, b)
            report.log_stat(f"mann_whitney_vol_by_sex_{col}", res)
            stat_results.append((f"vol_by_sex_{col}", res))

            with report.figure(f"vol_by_race_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df_race.select(Col.RACE, col).to_pandas(),
                    x=Col.RACE,
                    y=col,
                    order=RACE_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Race")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume by Race")
                ax.tick_params(axis="x", rotation=15)

            a = df_race.filter(pl.col(Col.RACE) == "White")[col].drop_nulls()
            b = df_race.filter(pl.col(Col.RACE) == "Black or African American")[col].drop_nulls()
            res = mann_whitney_result(a, b)
            report.log_stat(f"mann_whitney_vol_by_race_{col}", res)
            stat_results.append((f"vol_by_race_{col}", res))

            with report.figure(f"vol_by_age_{col}", figsize=(7, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select("age_bin", col).to_pandas(),
                    x="age_bin",
                    y=col,
                    order=AGE_BIN_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Age Bin")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume by Age Bin")

            groups = {
                bin_: df.filter(pl.col("age_bin") == bin_)[col].drop_nulls()
                for bin_ in AGE_BIN_ORDER
            }
            res = kruskal_result(groups)
            report.log_stat(f"kruskal_vol_by_age_{col}", res)
            stat_results.append((f"vol_by_age_{col}", res))

        # ------------------------------------------------------------------
        # 3. Voxel counts by scanner (resolution dependency check)
        # ------------------------------------------------------------------

        manufacturers = sorted(df[Col.MANUFACTURER].drop_nulls().unique().to_list())

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.N_VOXELS_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.N_VOXELS_DISC),
        ]:
            with report.figure(f"voxels_by_manufacturer_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.MANUFACTURER, col).to_pandas(),
                    x=Col.MANUFACTURER,
                    y=col,
                    ax=ax,
                )
                ax.set_xlabel("Manufacturer")
                ax.set_ylabel("Voxel count")
                ax.set_title(f"{label} Voxel Count by Manufacturer")
                ax.tick_params(axis="x", rotation=15)

            a = df.filter(pl.col(Col.MANUFACTURER) == manufacturers[0])[col].drop_nulls()
            b = df.filter(pl.col(Col.MANUFACTURER) == manufacturers[1])[col].drop_nulls()
            res = mann_whitney_result(a, b)
            report.log_stat(f"mann_whitney_voxels_by_manufacturer_{col}", res)
            stat_results.append((f"voxels_by_manufacturer_{col}", res))

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.VOLUME_MM3_DISC),
        ]:
            with report.figure(f"vol_mm3_by_manufacturer_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.MANUFACTURER, col).to_pandas(),
                    x=Col.MANUFACTURER,
                    y=col,
                    ax=ax,
                )
                ax.set_xlabel("Manufacturer")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume (mm³) by Manufacturer")
                ax.tick_params(axis="x", rotation=15)

            a = df.filter(pl.col(Col.MANUFACTURER) == manufacturers[0])[col].drop_nulls()
            b = df.filter(pl.col(Col.MANUFACTURER) == manufacturers[1])[col].drop_nulls()
            res = mann_whitney_result(a, b)
            report.log_stat(f"mann_whitney_vol_mm3_by_manufacturer_{col}", res)
            stat_results.append((f"vol_mm3_by_manufacturer_{col}", res))

        for col in [
            SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY,
            SegmentationVolumeCol.VOLUME_MM3_DISC,
        ]:
            mfr_stats = (
                df.group_by(Col.MANUFACTURER)
                .agg(
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).median().alias("median"),
                )
                .sort(Col.MANUFACTURER)
            )
            report.log_stat(f"{col}_by_manufacturer", mfr_stats.to_dicts())

        # ------------------------------------------------------------------
        # 4. Component counts by demographics (annotation completeness check)
        # ------------------------------------------------------------------

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.N_COMPONENTS_DISC),
        ]:
            with report.figure(f"comp_by_sex_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.SEX, col).to_pandas(),
                    x=Col.SEX,
                    y=col,
                    order=SEX_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Sex")
                ax.set_ylabel("Component count")
                ax.set_title(f"{label} Components by Sex")

            a = df.filter(pl.col(Col.SEX) == "Female")[col].drop_nulls()
            b = df.filter(pl.col(Col.SEX) == "Male")[col].drop_nulls()
            res = mann_whitney_result(a, b)
            report.log_stat(f"mann_whitney_comp_by_sex_{col}", res)
            stat_results.append((f"comp_by_sex_{col}", res))

            with report.figure(f"comp_by_race_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df_race.select(Col.RACE, col).to_pandas(),
                    x=Col.RACE,
                    y=col,
                    order=RACE_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Race")
                ax.set_ylabel("Component count")
                ax.set_title(f"{label} Components by Race")
                ax.tick_params(axis="x", rotation=15)

            a = df_race.filter(pl.col(Col.RACE) == "White")[col].drop_nulls()
            b = df_race.filter(pl.col(Col.RACE) == "Black or African American")[col].drop_nulls()
            res = mann_whitney_result(a, b)
            report.log_stat(f"mann_whitney_comp_by_race_{col}", res)
            stat_results.append((f"comp_by_race_{col}", res))

            with report.figure(f"comp_by_age_{col}", figsize=(7, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select("age_bin", col).to_pandas(),
                    x="age_bin",
                    y=col,
                    order=AGE_BIN_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Age Bin")
                ax.set_ylabel("Component count")
                ax.set_title(f"{label} Components by Age Bin")

            groups = {
                bin_: df.filter(pl.col("age_bin") == bin_)[col].drop_nulls()
                for bin_ in AGE_BIN_ORDER
            }
            res = kruskal_result(groups)
            report.log_stat(f"kruskal_comp_by_age_{col}", res)
            stat_results.append((f"comp_by_age_{col}", res))

        # ------------------------------------------------------------------
        # 5. BH-FDR correction across all omnibus p-values in this module
        # ------------------------------------------------------------------

        p_values = [r["p"] for _, r in stat_results]
        reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

        summary_rows = []
        for (key, r), p_a, rej in zip(stat_results, p_adj, reject):
            effect_name = "r_rb" if "r_rb" in r else "epsilon_sq"
            summary_rows.append(
                {
                    "stat_key": key,
                    "test": r["test"],
                    "p_raw": r["p"],
                    "p_adjusted": float(p_a),
                    "reject_h0": bool(rej),
                    "effect_size": r.get(effect_name),
                    "effect_name": effect_name,
                }
            )

        report.save_table(pl.DataFrame(summary_rows), "stats_summary")


if __name__ == "__main__":
    seg = load_segmentation_volumes()
    metadata = load_metadata()
    df = seg.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="left")
    run(df, "confounders")

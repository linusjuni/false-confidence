"""
Demographic grouping strategies for fairness analysis.

Provides configurable mappings from raw categorical/numeric values to
analysis groups. Each strategy is a named, immutable specification that
can be swapped by changing a single enum value.

Usage:
    from false_confidence.data.groups import race, RaceStrategy
    from false_confidence.data.schemas import Col

    spec = race[RaceStrategy.WHITE_VS_BLACK]
    df = spec.apply(df, Col.RACE)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from false_confidence.data.schemas import Col, Ethnicity, Race

# ---------------------------------------------------------------------------
# Grouping specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupingSpec:
    """Immutable specification for mapping raw values to analysis groups.

    For categorical columns, provide `mapping` (raw value → group label).
    For numeric columns, provide `binner` (a function that adds a group column).
    Rows whose raw values are not in the mapping are dropped.
    """

    description: str
    mapping: dict[str, str] | None = None
    binner: Callable[[pl.DataFrame, str], pl.DataFrame] | None = None
    group_col: str | None = None

    def apply(self, df: pl.DataFrame, col: str | Col) -> pl.DataFrame:
        """Apply this grouping to a DataFrame.

        Adds a column named `{col}_group` with the mapped/binned labels.
        Rows that don't map to any group are dropped.

        Args:
            df: Input DataFrame.
            col: Column to group on (Col enum or string).

        Returns:
            DataFrame with added group column, unmapped rows removed.
        """
        col = str(col)
        out_col = self.group_col or f"{col}_group"

        if self.mapping is not None:
            df = df.with_columns(
                pl.col(col).replace_strict(self.mapping, default=None).alias(out_col)
            )
            return df.filter(pl.col(out_col).is_not_null())

        if self.binner is not None:
            return self.binner(df, out_col)

        msg = "GroupingSpec must have either mapping or binner"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Strategy enums
# ---------------------------------------------------------------------------


class RaceStrategy(StrEnum):
    WHITE_VS_BLACK = "white_vs_black"
    WHITE_VS_BLACK_VS_OTHER = "white_vs_black_vs_other"
    WHITE_VS_NONWHITE = "white_vs_nonwhite"


class AgeStrategy(StrEnum):
    THREE_BINS = "three_bins"
    MEDIAN_SPLIT = "median_split"


class EthnicityStrategy(StrEnum):
    HISPANIC_VS_NOT = "hispanic_vs_not"


# ---------------------------------------------------------------------------
# Binner helpers (for numeric columns)
# ---------------------------------------------------------------------------


def _age_three_bins(df: pl.DataFrame, out_col: str) -> pl.DataFrame:
    """Bin age into <40 / 40-60 / 60+. Null ages (confirmed >89) map to 60+."""
    return df.with_columns(
        pl.when(pl.col(Col.AGE).is_null())
        .then(pl.lit("60+"))
        .when(pl.col(Col.AGE) < 40)
        .then(pl.lit("<40"))
        .when(pl.col(Col.AGE) < 60)
        .then(pl.lit("40-60"))
        .otherwise(pl.lit("60+"))
        .alias(out_col)
    )


def _age_median_split(df: pl.DataFrame, out_col: str) -> pl.DataFrame:
    """Split age at median into young/old. Drops rows with null age."""
    df = df.filter(pl.col(Col.AGE).is_not_null())
    median = df[Col.AGE].median()
    return df.with_columns(
        pl.when(pl.col(Col.AGE) < median)
        .then(pl.lit(f"<{median:.0f}"))
        .otherwise(pl.lit(f">={median:.0f}"))
        .alias(out_col)
    )


# ---------------------------------------------------------------------------
# Strategy registries
# ---------------------------------------------------------------------------

race: dict[RaceStrategy, GroupingSpec] = {
    RaceStrategy.WHITE_VS_BLACK: GroupingSpec(
        description="Primary analysis: two largest racial groups",
        mapping={
            Race.WHITE: "White",
            Race.BLACK: "Black",
        },
    ),
    RaceStrategy.WHITE_VS_BLACK_VS_OTHER: GroupingSpec(
        description="Three groups: White, Black, and all others merged",
        mapping={
            Race.WHITE: "White",
            Race.BLACK: "Black",
            Race.ASIAN: "Other",
            Race.AMERICAN_INDIAN: "Other",
            Race.PACIFIC_ISLANDER: "Other",
            Race.OTHER: "Other",
            Race.NOT_REPORTED: "Other",
        },
    ),
    RaceStrategy.WHITE_VS_NONWHITE: GroupingSpec(
        description="Binary: White vs. all non-White groups",
        mapping={
            Race.WHITE: "White",
            Race.BLACK: "Non-White",
            Race.ASIAN: "Non-White",
            Race.AMERICAN_INDIAN: "Non-White",
            Race.PACIFIC_ISLANDER: "Non-White",
            Race.OTHER: "Non-White",
            Race.NOT_REPORTED: "Non-White",
        },
    ),
}

age: dict[AgeStrategy, GroupingSpec] = {
    AgeStrategy.THREE_BINS: GroupingSpec(
        description="Three age bins: <40, 40-60, 60+",
        binner=_age_three_bins,
    ),
    AgeStrategy.MEDIAN_SPLIT: GroupingSpec(
        description="Binary split at median age",
        binner=_age_median_split,
    ),
}

ethnicity: dict[EthnicityStrategy, GroupingSpec] = {
    EthnicityStrategy.HISPANIC_VS_NOT: GroupingSpec(
        description="Hispanic vs. Non-Hispanic (drops Not Reported)",
        mapping={
            Ethnicity.HISPANIC: "Hispanic",
            Ethnicity.NOT_HISPANIC: "Non-Hispanic",
        },
    ),
}

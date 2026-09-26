"""Pure fairness metric functions.

All functions take a Polars DataFrame + column names and return
JSON-serializable dicts or floats. No I/O, no matplotlib.

Convention: score_col is a performance metric (higher = better for Dice and
nDSC, lower = better for HD95). DPD and DIR follow the canonical fairness
definition (Parikh et al.; Fairlearn; EEOC four-fifths rule): each case is
binarized into a *beneficial outcome* ("success") at a threshold, a per-group
success *rate* is computed, and then

    DPD = max(rate) - min(rate)        (absolute gap in success rate)
    DIR = min(rate) / max(rate)        (relative gap; 1.0 = parity)

The beneficial outcome is ``score > threshold`` when higher_is_better (Dice,
nDSC) and ``score < threshold`` otherwise (HD95). A DIR below 0.80 flags
adverse impact under the four-fifths rule (valid because DIR is a rate ratio).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from false_confidence.stats import kruskal_result, mann_whitney_result

# ---------------------------------------------------------------------------
# Group summary
# ---------------------------------------------------------------------------


def group_summary(df: pl.DataFrame, score_col: str, group_col: str) -> pl.DataFrame:
    """Per-group descriptive statistics, sorted by group name."""
    return (
        df.group_by(group_col)
        .agg(
            pl.col(score_col).count().alias("n"),
            pl.col(score_col).mean().alias("mean"),
            pl.col(score_col).median().alias("median"),
            pl.col(score_col).std().alias("std"),
            pl.col(score_col).quantile(0.25).alias("q25"),
            pl.col(score_col).quantile(0.75).alias("q75"),
        )
        .with_columns((pl.col("q75") - pl.col("q25")).alias("iqr"))
        .rename({group_col: "group"})
        .sort("group")
    )


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------


def _group_rates(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    threshold: float,
    higher_is_better: bool,
) -> dict[str, float]:
    """Per-group rate of the beneficial outcome, dropping NaN scores.

    The beneficial outcome ("success") is ``score > threshold`` when
    higher_is_better (Dice, nDSC) and ``score < threshold`` otherwise (HD95,
    lower-is-better). Returns the fraction of valid cases per group that succeed.
    """
    clean = df.filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_not_nan())
    success = pl.col(score_col) > threshold if higher_is_better else pl.col(score_col) < threshold
    rows = (
        clean.with_columns(success.cast(pl.Float64).alias("_success"))
        .group_by(group_col)
        .agg(pl.col("_success").mean().alias("rate"))
        .to_dicts()
    )
    if not rows:
        msg = f"No valid scores in {score_col} for any group"
        raise ValueError(msg)
    return {r[group_col]: float(r["rate"]) for r in rows}


def disparate_impact_ratio(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    threshold: float = 0.8,
    higher_is_better: bool = True,
) -> float:
    """DIR = min(group success rate) / max(group success rate). Range [0, 1]; 1.0 = parity.

    Success is binarized at ``threshold`` (Parikh et al. / four-fifths rule).
    Returns NaN if the best group's success rate is 0.
    """
    rates = _group_rates(df, score_col, group_col, threshold, higher_is_better)
    best = max(rates.values())
    worst = min(rates.values())
    if best == 0.0:
        return float("nan")
    return float(worst / best)


def demographic_parity_difference(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    threshold: float = 0.8,
    higher_is_better: bool = True,
) -> float:
    """DPD = max(rate) - min(rate): absolute gap in beneficial-outcome rate. Non-negative."""
    rates = _group_rates(df, score_col, group_col, threshold, higher_is_better)
    return float(max(rates.values()) - min(rates.values()))


def fairness_gap(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    threshold: float = 0.8,
    higher_is_better: bool = True,
) -> dict:
    """Bundle DIR, DPD, success rates, and group identities into a single dict."""
    rates = _group_rates(df, score_col, group_col, threshold, higher_is_better)
    best_group = max(rates, key=rates.get)  # type: ignore[arg-type]
    worst_group = min(rates, key=rates.get)  # type: ignore[arg-type]
    best_rate = rates[best_group]
    worst_rate = rates[worst_group]

    dir_val = worst_rate / best_rate if best_rate != 0.0 else float("nan")

    return {
        "dir": float(dir_val),
        "dpd": float(best_rate - worst_rate),
        "best_group": best_group,
        "worst_group": worst_group,
        "best_rate": float(best_rate),
        "worst_rate": float(worst_rate),
        "threshold": float(threshold),
        "higher_is_better": bool(higher_is_better),
        "n_groups": len(rates),
    }


def dir_sensitivity(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    thresholds: list[float],
    higher_is_better: bool = True,
) -> pl.DataFrame:
    """DIR/DPD across a grid of beneficial-outcome thresholds (sensitivity sweep).

    Guards against a near-degenerate cutoff: with high Dice (~0.89) the success
    rate at 0.8 can be near 1.0, so reviewers should confirm the verdict holds
    across thresholds.
    """
    rows = []
    for t in thresholds:
        gap = fairness_gap(df, score_col, group_col, threshold=t, higher_is_better=higher_is_better)
        rows.append(
            {
                "threshold": float(t),
                "dir": gap["dir"],
                "dpd": gap["dpd"],
                "best_group": gap["best_group"],
                "worst_group": gap["worst_group"],
                "best_rate": gap["best_rate"],
                "worst_rate": gap["worst_rate"],
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests (wrappers around false_confidence.stats)
# ---------------------------------------------------------------------------


def _extract_groups(df: pl.DataFrame, score_col: str, group_col: str) -> dict[str, np.ndarray]:
    """Extract per-group score arrays, dropping NaN values."""
    clean = df.filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_not_nan())
    groups: dict[str, np.ndarray] = {}
    for name in sorted(clean[group_col].unique().to_list()):
        arr = clean.filter(pl.col(group_col) == name)[score_col].to_numpy()
        if len(arr) > 0:
            groups[name] = arr
    return groups


def mann_whitney_test(df: pl.DataFrame, score_col: str, group_col: str) -> dict:
    """Mann-Whitney U for exactly two groups. Delegates to false_confidence.stats."""
    groups = _extract_groups(df, score_col, group_col)
    names = sorted(groups.keys())
    if len(names) != 2:
        msg = f"Expected 2 groups for Mann-Whitney, got {len(names)}: {names}"
        raise ValueError(msg)
    result = mann_whitney_result(groups[names[0]], groups[names[1]])
    result["group_a"] = names[0]
    result["group_b"] = names[1]
    return result


def kruskal_wallis_test(df: pl.DataFrame, score_col: str, group_col: str) -> dict:
    """Kruskal-Wallis H for 3+ groups. Delegates to false_confidence.stats."""
    groups = _extract_groups(df, score_col, group_col)
    if len(groups) < 3:
        msg = f"Expected 3+ groups for Kruskal-Wallis, got {len(groups)}"
        raise ValueError(msg)
    return kruskal_result(groups)


def apply_fdr(p_values: list[float], method: str = "fdr_bh") -> list[float]:
    """BH-FDR correction. Returns adjusted p-values."""
    if not p_values:
        return []
    _, corrected, _, _ = multipletests(p_values, method=method)
    return [float(p) for p in corrected]


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def ols_regression(df: pl.DataFrame, score_col: str, covariates: list[str]) -> dict:
    """OLS regression of score_col on covariates.

    Categorical covariates (string dtype) are one-hot encoded.
    Returns coefficients, R-squared, F-stat, and per-coefficient CIs.
    """
    import statsmodels.api as sm

    clean = df.select([score_col, *covariates]).drop_nulls().drop_nans()
    pdf = clean.to_pandas()

    y = pdf[score_col]
    X_parts = []
    for cov in covariates:
        if not __import__("pandas").api.types.is_numeric_dtype(pdf[cov]):
            dummies = pdf[[cov]].astype(str)
            dummies = __import__("pandas").get_dummies(dummies, drop_first=True, dtype=float)
            X_parts.append(dummies)
        else:
            X_parts.append(pdf[[cov]].astype(float))

    X = __import__("pandas").concat(X_parts, axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    coefficients = {}
    for name in model.params.index:
        ci = model.conf_int().loc[name]
        coefficients[name] = {
            "coef": float(model.params[name]),
            "se": float(model.bse[name]),
            "p": float(model.pvalues[name]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
        }

    return {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": coefficients,
        "n": int(model.nobs),
    }


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    metric_fn: Callable[[pl.DataFrame, str, str], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """BCa bootstrap confidence interval for a fairness metric.

    Uses scipy.stats.bootstrap (BCa method). Falls back to percentile
    if BCa returns NaN (degenerate distributions).
    """
    clean = df.filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_not_nan())
    n = clean.height
    rng = np.random.default_rng(seed)

    def _statistic(indices: np.ndarray) -> float:
        sample = clean[indices.astype(int).tolist()]
        return metric_fn(sample, score_col, group_col)

    point_estimate = metric_fn(clean, score_col, group_col)

    indices = (np.arange(n),)
    try:
        result = sp_stats.bootstrap(
            indices,
            statistic=_statistic,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method="BCa",
            random_state=rng,
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)
        method = "bca"
    except Exception:
        result = sp_stats.bootstrap(
            indices,
            statistic=_statistic,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method="percentile",
            random_state=rng,
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)
        method = "percentile"

    if np.isnan(ci_low) or np.isnan(ci_high):
        result = sp_stats.bootstrap(
            indices,
            statistic=_statistic,
            n_resamples=n_boot,
            confidence_level=1 - alpha,
            method="percentile",
            random_state=rng,
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)
        method = "percentile"

    return {
        "point_estimate": float(point_estimate),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "n_boot": n_boot,
        "method": method,
    }


def permutation_test(
    df: pl.DataFrame,
    score_col: str,
    group_col: str,
    metric_fn: Callable[[pl.DataFrame, str, str], float],
    n_perm: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Permutation test: is the observed metric significantly different from chance?

    Shuffles group labels n_perm times, computes metric_fn each time.
    """
    clean = df.filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_not_nan())
    rng = np.random.default_rng(seed)

    observed = metric_fn(clean, score_col, group_col)

    group_values = clean[group_col].to_numpy()
    null_dist = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(group_values)
        permuted = clean.with_columns(pl.Series(group_col, shuffled))
        null_dist[i] = metric_fn(permuted, score_col, group_col)

    p_value = float(
        np.mean(
            np.abs(null_dist - np.nanmean(null_dist)) >= np.abs(observed - np.nanmean(null_dist))
        )
    )

    return {
        "observed": float(observed),
        "p_value": p_value,
        "n_perm": n_perm,
        "null_mean": float(np.nanmean(null_dist)),
        "null_std": float(np.nanstd(null_dist)),
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def dir_widening(dir_gold: float, dir_silver: float) -> dict:
    """Compute % widening of DIR between gold and silver rulers.

    Positive widening_pct means the silver ruler exaggerates the gap.
    """
    gap_gold = 1.0 - dir_gold
    gap_silver = 1.0 - dir_silver

    if gap_gold == 0.0:
        widening_pct = float("inf") if gap_silver != 0.0 else 0.0
    else:
        widening_pct = (gap_silver - gap_gold) / gap_gold * 100.0

    if widening_pct > 0:
        direction = "widened"
    elif widening_pct < 0:
        direction = "narrowed"
    else:
        direction = "unchanged"

    return {
        "dir_gold": float(dir_gold),
        "dir_silver": float(dir_silver),
        "widening_pct": float(widening_pct),
        "direction": direction,
    }


def compare_fairness_gaps(gaps: list[dict], labels: list[str]) -> pl.DataFrame:
    """Side-by-side comparison table of fairness gaps across rulers or models."""
    rows = []
    for gap, label in zip(gaps, labels):
        rows.append(
            {
                "label": label,
                "dir": gap["dir"],
                "dpd": gap["dpd"],
                "best_group": gap["best_group"],
                "worst_group": gap["worst_group"],
                "best_rate": gap["best_rate"],
                "worst_rate": gap["worst_rate"],
            }
        )
    return pl.DataFrame(rows)

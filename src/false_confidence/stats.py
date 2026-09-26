"""
Statistical test utilities for EDA modules.

Each function takes raw data and returns a serializable dict.
Callers are responsible for dropping nulls before passing data in.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def mann_whitney_result(a: Any, b: Any) -> dict:
    """
    Mann-Whitney U test for two independent groups.

    Parameters
    ----------
    a, b : array-like
        Observations for group A and group B (nulls already dropped).

    Returns
    -------
    dict with keys: test, U, p, r_rb, median_a, median_b, n_a, n_b
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # scipy.stats.mannwhitneyu raises ValueError when every observation across
    # both groups is identical (no ranks to compare). This is a real case for
    # the silver ruler, where HD95 between two near-identical predictions is
    # 0.0 for a whole group. Treat it as "no detectable difference" (p=1.0).
    if np.unique(np.concatenate([a, b])).size <= 1:
        U, p, r_rb = float("nan"), 1.0, 0.0
    else:
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r_rb = 1.0 - (2.0 * U) / (len(a) * len(b))
    return {
        "test": "mann_whitney",
        "U": float(U),
        "p": float(p),
        "r_rb": float(r_rb),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def kruskal_result(groups: dict[str, Any]) -> dict:
    """
    Kruskal-Wallis H test for three or more independent groups, with
    Dunn's post-hoc (Bonferroni) if the omnibus test is significant.

    Parameters
    ----------
    groups : dict mapping group label -> array-like
        Observations per group (nulls already dropped).

    Returns
    -------
    dict with keys: test, H, p, epsilon_sq, posthoc, medians, ns
    """
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    labels = list(groups.keys())
    k = len(arrays)
    N = sum(len(a) for a in arrays)

    # stats.kruskal raises ValueError when every observation across all groups
    # is identical (e.g. silver-ruler HD95 all 0.0). Treat as no difference.
    if np.unique(np.concatenate(arrays)).size <= 1:
        H, p, epsilon_sq = float("nan"), 1.0, 0.0
    else:
        H, p = stats.kruskal(*arrays)
        epsilon_sq = (H - k + 1) / (N - k)

    posthoc_pairs: list[dict] = []
    if p < 0.05:
        ph = sp.posthoc_dunn(arrays, p_adjust="bonferroni")
        # ph is a symmetric k×k DataFrame indexed 1..k
        for i in range(k):
            for j in range(i + 1, k):
                posthoc_pairs.append(
                    {
                        "pair": f"{labels[i]} vs {labels[j]}",
                        "p": float(ph.iloc[i, j]),
                    }
                )

    return {
        "test": "kruskal_wallis",
        "H": float(H),
        "p": float(p),
        "epsilon_sq": float(epsilon_sq),
        "posthoc": posthoc_pairs,
        "medians": {label: float(np.median(a)) for label, a in zip(labels, arrays)},
        "ns": {label: int(len(a)) for label, a in zip(labels, arrays)},
    }


def chi2_result(contingency: pd.DataFrame) -> dict:
    """
    Chi-squared test of independence plus Cramér's V effect size.

    Parameters
    ----------
    contingency : pd.DataFrame
        Contingency table with count values (rows and columns are category labels).

    Returns
    -------
    dict with keys: test, chi2, p, dof, cramers_v
    """
    chi2, p, dof, _ = stats.chi2_contingency(contingency.values)
    N = contingency.values.sum()
    r, c = contingency.shape
    cramers_v = math.sqrt(chi2 / (N * (min(r, c) - 1)))
    return {
        "test": "chi2",
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
    }

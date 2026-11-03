"""Multiple-testing corrections for pair screening p-values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.multitest import multipletests

from hft_research.exceptions import PairDiscoveryError


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """Corrected p-values and rejection mask."""

    method: str
    raw_pvalues: np.ndarray
    corrected_pvalues: np.ndarray
    rejected: np.ndarray
    alpha: float


_METHOD_MAP = {
    "bonferroni": "bonferroni",
    "benjamini_hochberg": "fdr_bh",
    "fdr_bh": "fdr_bh",
    "none": "none",
}


def correct_pvalues(
    pvalues: np.ndarray,
    *,
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Apply FWER/FDR correction across a family of hypothesis tests.

    Parameters
    ----------
    pvalues:
        Array of raw p-values from independent (or positively dependent)
        cointegration / stationarity tests.
    method:
        ``bonferroni``, ``benjamini_hochberg`` / ``fdr_bh``, or ``none``.
    alpha:
        Family-wise or FDR significance level.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    if p.size == 0:
        return MultipleTestingResult(
            method=method,
            raw_pvalues=p,
            corrected_pvalues=p.copy(),
            rejected=np.array([], dtype=bool),
            alpha=alpha,
        )
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise PairDiscoveryError("p-values must be finite and in [0, 1]")
    if alpha <= 0 or alpha >= 1:
        raise PairDiscoveryError("alpha must be in (0, 1)")

    key = method.lower().strip()
    if key not in _METHOD_MAP:
        raise PairDiscoveryError(f"Unknown multiple-testing method: {method!r}")

    mapped = _METHOD_MAP[key]
    if mapped == "none":
        return MultipleTestingResult(
            method="none",
            raw_pvalues=p,
            corrected_pvalues=p.copy(),
            rejected=p < alpha,
            alpha=alpha,
        )

    rejected, corrected, _, _ = multipletests(p, alpha=alpha, method=mapped)
    return MultipleTestingResult(
        method=key,
        raw_pvalues=p,
        corrected_pvalues=np.asarray(corrected, dtype=np.float64),
        rejected=np.asarray(rejected, dtype=bool),
        alpha=alpha,
    )

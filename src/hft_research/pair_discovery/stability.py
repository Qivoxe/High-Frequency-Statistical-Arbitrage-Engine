"""Rolling cointegration stability analysis (causal windows only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_research.exceptions import InsufficientDataError, PairDiscoveryError
from hft_research.logging_utils import get_logger
from hft_research.pair_discovery.engle_granger import engle_granger_test

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RollingStabilityResult:
    """Rolling Engle–Granger stability diagnostics."""

    window: int
    step: int
    n_windows: int
    pvalues: np.ndarray
    hedge_ratios: np.ndarray
    half_lives: np.ndarray
    cointegrated_mask: np.ndarray
    stability_ratio: float
    hedge_ratio_std: float
    hedge_ratio_cv: float
    end_indices: np.ndarray


def rolling_cointegration_stability(
    y: np.ndarray,
    x: np.ndarray,
    *,
    window: int = 252,
    step: int = 21,
    significance: float = 0.05,
) -> RollingStabilityResult:
    """Evaluate Engle–Granger on contiguous rolling windows.

    Windows are strictly causal: window ``i`` uses observations
    ``[end - window, end)`` where ``end`` advances by ``step``. No future
    data beyond ``end`` enters the estimate (no look-ahead).
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    if y_arr.size != x_arr.size:
        raise PairDiscoveryError("y and x length mismatch")
    if window < 30:
        raise PairDiscoveryError("rolling window must be >= 30")
    if step < 1:
        raise PairDiscoveryError("step must be >= 1")
    n = y_arr.size
    if n < window:
        raise InsufficientDataError(
            f"Need >= {window} observations for rolling stability, got {n}"
        )

    ends = list(range(window, n + 1, step))
    pvals: list[float] = []
    ratios: list[float] = []
    hls: list[float] = []
    masks: list[bool] = []

    for end in ends:
        start = end - window
        # Causal slice — only past data relative to `end`
        res = engle_granger_test(
            y_arr[start:end],
            x_arr[start:end],
            significance=significance,
        )
        pvals.append(res.residual_adf.pvalue)
        ratios.append(res.hedge_ratio)
        hls.append(res.half_life)
        masks.append(res.is_cointegrated)

    p_arr = np.asarray(pvals, dtype=np.float64)
    r_arr = np.asarray(ratios, dtype=np.float64)
    hl_arr = np.asarray(hls, dtype=np.float64)
    m_arr = np.asarray(masks, dtype=bool)
    stability = float(np.mean(m_arr)) if m_arr.size else 0.0
    hr_std = float(np.std(r_arr, ddof=1)) if r_arr.size > 1 else 0.0
    hr_mean = float(np.mean(np.abs(r_arr))) if r_arr.size else 1.0
    hr_cv = hr_std / hr_mean if hr_mean > 1e-12 else float("inf")

    logger.debug(
        "Rolling stability: windows=%d ratio=%.3f hr_cv=%.4f",
        len(ends),
        stability,
        hr_cv,
    )
    return RollingStabilityResult(
        window=window,
        step=step,
        n_windows=len(ends),
        pvalues=p_arr,
        hedge_ratios=r_arr,
        half_lives=hl_arr,
        cointegrated_mask=m_arr,
        stability_ratio=stability,
        hedge_ratio_std=hr_std,
        hedge_ratio_cv=hr_cv,
        end_indices=np.asarray(ends, dtype=np.int64),
    )

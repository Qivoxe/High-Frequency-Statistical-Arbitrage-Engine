"""Engle–Granger two-step cointegration test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from hft_research.exceptions import InsufficientDataError, PairDiscoveryError
from hft_research.pair_discovery.stationarity import ADFResult, adf_test, half_life_ou


@dataclass(frozen=True, slots=True)
class EngleGrangerResult:
    """Result of the Engle–Granger cointegration procedure."""

    hedge_ratio: float
    intercept: float
    residual_adf: ADFResult
    is_cointegrated: bool
    half_life: float
    residual_std: float
    nobs: int
    residuals: np.ndarray


def engle_granger_test(
    y: np.ndarray,
    x: np.ndarray,
    *,
    significance: float = 0.05,
    add_intercept: bool = True,
) -> EngleGrangerResult:
    """Engle–Granger: OLS of ``y`` on ``x``, then ADF on residuals.

    The cointegrating vector is estimated on the *full* provided sample.
    Callers must pass only in-sample / training windows to avoid look-ahead
    when screening pairs for live trading.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    if y_arr.size != x_arr.size:
        raise PairDiscoveryError("y and x must have equal length")
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr = y_arr[mask]
    x_arr = x_arr[mask]
    if y_arr.size < 30:
        raise InsufficientDataError(
            f"Engle-Granger requires >= 30 finite observations, got {y_arr.size}"
        )

    try:
        if add_intercept:
            X = add_constant(x_arr)
            model = OLS(y_arr, X).fit()
            intercept = float(model.params[0])
            hedge_ratio = float(model.params[1])
        else:
            model = OLS(y_arr, x_arr).fit()
            intercept = 0.0
            hedge_ratio = float(model.params[0])
        residuals = np.asarray(model.resid, dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        raise PairDiscoveryError(f"Engle-Granger OLS failed: {exc}") from exc

    adf = adf_test(residuals, significance=significance)
    try:
        hl = half_life_ou(residuals)
    except InsufficientDataError:
        hl = float("inf")

    return EngleGrangerResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        residual_adf=adf,
        is_cointegrated=adf.is_stationary,
        half_life=hl,
        residual_std=float(np.std(residuals, ddof=1)),
        nobs=int(y_arr.size),
        residuals=residuals,
    )

"""ADF stationarity tests and Ornstein–Uhlenbeck half-life estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller

from hft_research.exceptions import InsufficientDataError, PairDiscoveryError


@dataclass(frozen=True, slots=True)
class ADFResult:
    """Augmented Dickey–Fuller test result."""

    statistic: float
    pvalue: float
    used_lag: int
    nobs: int
    critical_values: dict[str, float]
    is_stationary: bool


def adf_test(
    series: np.ndarray,
    *,
    significance: float = 0.05,
    maxlag: int | None = None,
    regression: str = "c",
    autolag: str | None = "AIC",
) -> ADFResult:
    """Run the Augmented Dickey–Fuller unit-root test.

    Null hypothesis: series has a unit root (non-stationary).
    Rejecting the null (pvalue < significance) implies stationarity.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 20:
        raise InsufficientDataError(f"ADF requires >= 20 observations, got {x.size}")
    try:
        stat, pvalue, usedlag, nobs, crit, _ = adfuller(
            x,
            maxlag=maxlag,
            regression=regression,
            autolag=autolag,
        )
    except Exception as exc:  # noqa: BLE001 — wrap library errors
        raise PairDiscoveryError(f"ADF failed: {exc}") from exc
    return ADFResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        used_lag=int(usedlag),
        nobs=int(nobs),
        critical_values={str(k): float(v) for k, v in crit.items()},
        is_stationary=bool(pvalue < significance),
    )


def half_life_ou(spread: np.ndarray) -> float:
    """Estimate Ornstein–Uhlenbeck mean-reversion half-life of a spread.

    Fits ``Δs_t = λ s_{t-1} + ε`` via OLS. Half-life = ``-ln(2) / λ``.
    Returns ``np.inf`` if the process is not mean-reverting (λ >= 0).
    """
    s = np.asarray(spread, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 20:
        raise InsufficientDataError(f"Half-life requires >= 20 observations, got {s.size}")
    lag = s[:-1]
    delta = np.diff(s)
    # OLS: delta = lambda * lag  (no intercept for pure OU speed; include const)
    X = np.column_stack([np.ones(lag.size), lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, delta, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise PairDiscoveryError(f"Half-life OLS failed: {exc}") from exc
    lam = float(beta[1])
    # Non-mean-reverting or numerically degenerate λ → infinite half-life
    if lam >= -1e-12:
        return float("inf")
    hl = float(-np.log(2.0) / lam)
    if not np.isfinite(hl) or hl > 1e6:
        return float("inf")
    return hl

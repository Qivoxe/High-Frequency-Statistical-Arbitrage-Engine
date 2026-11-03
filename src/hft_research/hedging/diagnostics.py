"""Kalman filter diagnostics: whiteness, NEES, divergence summary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from hft_research.config import HedgingConfig
from hft_research.exceptions import KalmanFilterError
from hft_research.hedging.kalman import KalmanHedgeFilter, KalmanState


@dataclass(frozen=True, slots=True)
class FilterDiagnostics:
    """Summary diagnostics for a completed filter run."""

    nobs: int
    innovation_mean: float
    innovation_std: float
    standardized_innovation_mean: float
    standardized_innovation_std: float
    ljung_box_pvalue: float
    innovations_white: bool
    mean_nees: float | None
    nees_ok: bool | None
    diverged: bool
    max_abs_z: float
    final_alpha: float
    final_beta: float


def _ljung_box_pvalue(innovations: np.ndarray, lags: int = 10) -> float:
    """Portmanteau Ljung–Box test on standardized innovations."""
    x = np.asarray(innovations, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    if n < lags + 5:
        return float("nan")
    acf = np.correlate(x, x, mode="full")[n - 1 :]
    acf = acf / acf[0]
    q = n * (n + 2) * np.sum((acf[1 : lags + 1] ** 2) / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(q, df=lags))


def diagnose_filter(
    filt: KalmanHedgeFilter,
    *,
    true_states: np.ndarray | None = None,
    config: HedgingConfig | None = None,
    whiteness_alpha: float = 0.01,
) -> FilterDiagnostics:
    """Compute filter health metrics from history.

    Parameters
    ----------
    filt:
        Filter with populated history.
    true_states:
        Optional ``(T, 2)`` true ``[alpha, beta]`` for NEES evaluation
        (synthetic experiments only — never available in production).
    config:
        Hedging config for NEES threshold; defaults to ``filt.config``.
    whiteness_alpha:
        Significance for rejecting innovation whiteness.
    """
    history: list[KalmanState] = filt.history
    if not history:
        raise KalmanFilterError("Cannot diagnose empty filter history")
    cfg = config or filt.config

    nus = np.array([s.innovation for s in history], dtype=np.float64)
    zs = np.array([s.z_score for s in history], dtype=np.float64)
    lb_p = _ljung_box_pvalue(zs)

    mean_nees: float | None = None
    nees_ok: bool | None = None
    if true_states is not None:
        truth = np.asarray(true_states, dtype=np.float64)
        if truth.shape[0] != len(history) or truth.shape[1] != 2:
            raise KalmanFilterError("true_states must have shape (T, 2) matching history")
        nees_vals: list[float] = []
        for s, th in zip(history, truth, strict=True):
            err = np.array([s.alpha, s.beta]) - th
            try:
                invP = np.linalg.inv(s.P)
            except np.linalg.LinAlgError as exc:
                raise KalmanFilterError("Singular P in NEES") from exc
            nees_vals.append(float(err @ invP @ err))
        mean_nees = float(np.mean(nees_vals))
        nees_ok = mean_nees <= cfg.nees_upper

    return FilterDiagnostics(
        nobs=len(history),
        innovation_mean=float(np.mean(nus)),
        innovation_std=float(np.std(nus, ddof=1)),
        standardized_innovation_mean=float(np.mean(zs)),
        standardized_innovation_std=float(np.std(zs, ddof=1)),
        ljung_box_pvalue=lb_p,
        innovations_white=bool(np.isnan(lb_p) or lb_p >= whiteness_alpha),
        mean_nees=mean_nees,
        nees_ok=nees_ok,
        diverged=filt.diverged,
        max_abs_z=float(np.max(np.abs(zs))),
        final_alpha=history[-1].alpha,
        final_beta=history[-1].beta,
    )

"""Johansen cointegration tests (trace and maximum eigenvalue)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from hft_research.exceptions import InsufficientDataError, PairDiscoveryError

# Map config strings to statsmodels det_order
_DET_ORDER = {
    "none": -1,
    "const": 0,
    "trend": 1,
}

# Critical value column indices in statsmodels for 90%, 95%, 99%
_SIG_TO_COL = {
    0.10: 0,
    0.05: 1,
    0.01: 2,
}


@dataclass(frozen=True, slots=True)
class JohansenResult:
    """Johansen trace and max-eigenvalue test results for a bivariate system."""

    trace_stat: np.ndarray
    trace_crit: np.ndarray
    max_eig_stat: np.ndarray
    max_eig_crit: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    cointegrating_vector: np.ndarray
    rank_trace: int
    rank_max_eig: int
    is_cointegrated: bool
    nobs: int
    k_ar_diff: int


def _critical_column(significance: float) -> int:
    # Snap to nearest supported level
    keys = sorted(_SIG_TO_COL.keys(), key=lambda a: abs(a - significance))
    return _SIG_TO_COL[keys[0]]


def johansen_test(
    data: np.ndarray,
    *,
    det_order: str = "const",
    k_ar_diff: int = 1,
    significance: float = 0.05,
) -> JohansenResult:
    """Johansen trace and maximum-eigenvalue tests.

    Parameters
    ----------
    data:
        Array of shape ``(T, k)`` with ``k >= 2`` series in columns.
        For pairs use ``k=2``.
    det_order:
        Deterministic term: ``none``, ``const``, or ``trend``.
    k_ar_diff:
        Number of lagged differences in the VECM.
    significance:
        Level used to decide cointegration rank (approx 10/5/1%).

    Notes
    -----
    The cointegrating vector is taken as the eigenvector corresponding to
    the largest eigenvalue, normalized so the first component equals 1
    (y-loading convention for pairs).
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise PairDiscoveryError("Johansen data must be 2-D (T, k)")
    if x.shape[1] < 2:
        raise PairDiscoveryError("Johansen requires at least 2 series")
    mask = np.all(np.isfinite(x), axis=1)
    x = x[mask]
    if x.shape[0] < 40:
        raise InsufficientDataError(
            f"Johansen requires >= 40 observations, got {x.shape[0]}"
        )
    if det_order not in _DET_ORDER:
        raise PairDiscoveryError(f"Unknown det_order: {det_order!r}")
    if k_ar_diff < 0:
        raise PairDiscoveryError("k_ar_diff must be >= 0")

    try:
        res = coint_johansen(x, det_order=_DET_ORDER[det_order], k_ar_diff=k_ar_diff)
    except Exception as exc:  # noqa: BLE001
        raise PairDiscoveryError(f"Johansen test failed: {exc}") from exc

    col = _critical_column(significance)
    trace_stat = np.asarray(res.lr1, dtype=np.float64)
    trace_crit = np.asarray(res.cvt[:, col], dtype=np.float64)
    max_stat = np.asarray(res.lr2, dtype=np.float64)
    max_crit = np.asarray(res.cvm[:, col], dtype=np.float64)

    # Rank: number of consecutive rejections starting from r=0
    rank_trace = int(np.sum(trace_stat > trace_crit))
    # More carefully: sequential from r=0
    rank_trace = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_crit[i]:
            rank_trace = i + 1
        else:
            break
    rank_max = 0
    for i in range(len(max_stat)):
        if max_stat[i] > max_crit[i]:
            rank_max = i + 1
        else:
            break

    evecs = np.asarray(res.evec, dtype=np.float64)
    # Largest eigenvalue eigenvector is column 0 in statsmodels
    cvec = evecs[:, 0].copy()
    if abs(cvec[0]) > 1e-15:
        cvec = cvec / cvec[0]

    return JohansenResult(
        trace_stat=trace_stat,
        trace_crit=trace_crit,
        max_eig_stat=max_stat,
        max_eig_crit=max_crit,
        eigenvalues=np.asarray(res.eig, dtype=np.float64),
        eigenvectors=evecs,
        cointegrating_vector=cvec,
        rank_trace=rank_trace,
        rank_max_eig=rank_max,
        is_cointegrated=rank_trace >= 1 or rank_max >= 1,
        nobs=int(x.shape[0]),
        k_ar_diff=k_ar_diff,
    )

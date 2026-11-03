"""Kalman hedge filter convergence and edge-case tests."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.config import HedgingConfig
from hft_research.exceptions import KalmanFilterError
from hft_research.hedging.diagnostics import diagnose_filter
from hft_research.hedging.kalman import KalmanHedgeFilter
from hft_research.synthetic.cointegrated import generate_cointegrated_pair


def test_kalman_converges_to_true_hedge() -> None:
    pair = generate_cointegrated_pair(seed=11, n=1500, hedge_ratio=1.25, intercept=0.5)
    cfg = HedgingConfig(
        observation_noise=1e-4,
        process_noise_diag=(1e-8, 1e-9),
        adaptive_process_noise=False,
        initial_P_diag=(100.0, 10.0),
    )
    filt = KalmanHedgeFilter(config=cfg)
    filt.filter_series(pair.y, pair.x, theta0=np.array([0.0, 0.0]))
    # After burn-in, estimates should be close
    assert filt.beta == pytest.approx(pair.hedge_ratio, rel=0.08)
    assert filt.alpha == pytest.approx(pair.intercept, abs=0.25)
    assert not filt.diverged


def test_kalman_tracks_time_varying_beta() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    x = 100 + np.cumsum(rng.normal(0, 0.01, size=n))
    beta = np.linspace(0.8, 1.4, n)
    alpha = 0.2
    y = alpha + beta * x + rng.normal(0, 0.002, size=n)
    cfg = HedgingConfig(
        observation_noise=1e-5,
        process_noise_diag=(1e-6, 1e-5),
        adaptive_process_noise=True,
        adaptive_kappa=0.1,
        initial_P_diag=(10.0, 1.0),
    )
    filt = KalmanHedgeFilter(config=cfg)
    states = filt.filter_series(y, x)
    # Last quarter average beta near true end value
    est = np.mean([s.beta for s in states[-200:]])
    assert est == pytest.approx(beta[-1], rel=0.15)


def test_diagnostics_nees() -> None:
    pair = generate_cointegrated_pair(seed=3, n=400)
    true = np.column_stack(
        [
            np.full(pair.y.size, pair.intercept),
            np.full(pair.y.size, pair.hedge_ratio),
        ]
    )
    cfg = HedgingConfig(
        observation_noise=1e-4,
        process_noise_diag=(1e-8, 1e-9),
        adaptive_process_noise=False,
    )
    filt = KalmanHedgeFilter(config=cfg)
    # Skip first few observations in NEES alignment: filter_series skips NaNs only
    states = filt.filter_series(pair.y, pair.x)
    diag = diagnose_filter(filt, true_states=true[: len(states)])
    assert diag.nobs == len(states)
    assert diag.mean_nees is not None
    assert diag.final_beta == pytest.approx(filt.beta)


def test_invalid_inputs() -> None:
    filt = KalmanHedgeFilter()
    with pytest.raises(KalmanFilterError):
        filt.update(float("nan"), 1.0)
    with pytest.raises(KalmanFilterError):
        filt.filter_series([1.0, 2.0], [1.0])
    with pytest.raises(KalmanFilterError):
        filt.reset(theta0=np.array([1.0, 2.0, 3.0]))


def test_divergence_flag() -> None:
    cfg = HedgingConfig(
        observation_noise=1e-8,
        process_noise_diag=(1e-12, 1e-12),
        adaptive_process_noise=False,
        divergence_z=1.0,
        divergence_streak=3,
        initial_P_diag=(1e-6, 1e-6),
    )
    filt = KalmanHedgeFilter(config=cfg)
    # Wildly inconsistent observations with tiny R → large z
    for i in range(5):
        filt.update(y=1000.0 * (i + 1), x=1.0)
    assert filt.diverged


def test_missing_observations_skipped() -> None:
    filt = KalmanHedgeFilter()
    y = np.array([1.0, np.nan, 1.1, 1.2])
    x = np.array([1.0, 1.0, 1.0, 1.0])
    states = filt.filter_series(y, x)
    assert len(states) == 3

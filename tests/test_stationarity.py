"""ADF and half-life unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.exceptions import InsufficientDataError
from hft_research.pair_discovery.stationarity import adf_test, half_life_ou


def test_adf_stationary_series() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    res = adf_test(x, significance=0.05)
    assert res.is_stationary
    assert res.pvalue < 0.05


def test_adf_random_walk_not_stationary() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=500))
    res = adf_test(x, significance=0.01)
    assert not res.is_stationary


def test_adf_insufficient_data() -> None:
    with pytest.raises(InsufficientDataError):
        adf_test(np.arange(5.0))


def test_half_life_ou_known() -> None:
    # AR(1) with phi=0.9 → kappa=0.1 → HL = ln2/0.1 ≈ 6.93
    rng = np.random.default_rng(2)
    n = 2000
    phi = 0.9
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, 0.01)
    hl = half_life_ou(s)
    assert 4.0 < hl < 12.0


def test_half_life_non_mean_reverting() -> None:
    x = np.cumsum(np.ones(100))
    assert half_life_ou(x) == float("inf")

"""Engle–Granger cointegration tests."""

from __future__ import annotations

import pytest

from hft_research.exceptions import InsufficientDataError, PairDiscoveryError
from hft_research.pair_discovery.engle_granger import engle_granger_test


def test_detects_cointegrated_pair(cointegrated_pair) -> None:
    res = engle_granger_test(cointegrated_pair.y, cointegrated_pair.x, significance=0.05)
    assert res.is_cointegrated
    assert res.hedge_ratio == pytest.approx(cointegrated_pair.hedge_ratio, rel=0.05)
    # Intercept OLS estimate is noisier on finite RW samples; allow wider band
    assert res.intercept == pytest.approx(cointegrated_pair.intercept, abs=0.75)
    assert res.half_life < 50


def test_rejects_independent_pair(independent_pair) -> None:
    res = engle_granger_test(independent_pair.y, independent_pair.x, significance=0.01)
    # Independent RWs should typically not cointegrate
    assert not res.is_cointegrated or res.residual_adf.pvalue > 0.01


def test_length_mismatch() -> None:
    with pytest.raises(PairDiscoveryError):
        engle_granger_test([1, 2, 3], [1, 2])


def test_insufficient() -> None:
    with pytest.raises(InsufficientDataError):
        engle_granger_test(range(10), range(10))

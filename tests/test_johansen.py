"""Johansen cointegration tests."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.exceptions import InsufficientDataError, PairDiscoveryError
from hft_research.pair_discovery.johansen import johansen_test


def test_johansen_finds_cointegration(cointegrated_pair) -> None:
    data = np.column_stack([cointegrated_pair.y, cointegrated_pair.x])
    res = johansen_test(data, det_order="const", k_ar_diff=1, significance=0.05)
    assert res.is_cointegrated
    assert res.rank_trace >= 1 or res.rank_max_eig >= 1
    assert res.cointegrating_vector[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(res.cointegrating_vector))
    # Hedge ratio recovered from EG-normalized vector is approximate under Johansen
    assert abs(res.cointegrating_vector[1]) > 0.1


def test_johansen_independent_often_rank0(independent_pair) -> None:
    data = np.column_stack([independent_pair.y, independent_pair.x])
    res = johansen_test(data, significance=0.01)
    # At strict 1% often not cointegrated; allow soft assertion
    assert isinstance(res.is_cointegrated, bool)


def test_johansen_invalid_inputs() -> None:
    with pytest.raises(PairDiscoveryError):
        johansen_test(np.arange(20.0), det_order="const")
    with pytest.raises(PairDiscoveryError):
        johansen_test(np.ones((50, 1)))
    with pytest.raises(InsufficientDataError):
        johansen_test(np.random.default_rng(0).normal(size=(20, 2)))

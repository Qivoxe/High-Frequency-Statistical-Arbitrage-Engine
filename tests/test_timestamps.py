"""Tests for timestamp normalization."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.exceptions import TimestampError
from hft_research.market_data.timestamps import (
    TimestampNormalizer,
    normalize_timestamps,
    unit_to_ns_factor,
)


def test_unit_factors() -> None:
    assert unit_to_ns_factor("ns") == 1
    assert unit_to_ns_factor("us") == 1_000
    assert unit_to_ns_factor("ms") == 1_000_000
    assert unit_to_ns_factor("s") == 1_000_000_000


def test_normalize_ms_to_ns() -> None:
    out = normalize_timestamps([1, 2, 3], unit="ms")
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, np.array([1_000_000, 2_000_000, 3_000_000]))


def test_reject_nan_and_negative() -> None:
    with pytest.raises(TimestampError):
        normalize_timestamps([1.0, np.nan])
    with pytest.raises(TimestampError):
        normalize_timestamps([-1])


def test_out_of_order_drop() -> None:
    norm = TimestampNormalizer(unit="ns", max_out_of_order_ns=10)
    assert norm.normalize_one(100) == 100
    assert norm.normalize_one(50) is None  # lag 50 > 10
    assert norm.dropped_count == 1
    assert norm.normalize_one(110) == 110


def test_unknown_unit() -> None:
    with pytest.raises(TimestampError):
        unit_to_ns_factor("hours")

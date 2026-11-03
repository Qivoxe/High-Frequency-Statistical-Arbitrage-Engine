"""Trade aggregation and volume-bar tests."""

from __future__ import annotations

import pytest

from hft_research.exceptions import MarketDataError
from hft_research.market_data.bars import VolumeBarAggregator, aggregate_volume_bars
from hft_research.market_data.trades import Trade, TradeAggregator


def _trades() -> list[Trade]:
    return [
        Trade(1, 100.0, 40.0, 1, 0),
        Trade(2, 101.0, 40.0, -1, 1),
        Trade(3, 102.0, 40.0, 1, 2),
        Trade(4, 103.0, 30.0, -1, 3),
    ]


def test_trade_aggregator_vwap() -> None:
    agg = TradeAggregator()
    agg.add_many(_trades())
    expected = (100 * 40 + 101 * 40 + 102 * 40 + 103 * 30) / 150
    assert agg.vwap() == pytest.approx(expected)
    df = agg.to_frame()
    assert df.height == 4
    assert df["ts"].is_sorted()


def test_invalid_trade() -> None:
    with pytest.raises(MarketDataError):
        TradeAggregator().add(Trade(1, 100.0, 0.0))


def test_volume_bars_deterministic_split() -> None:
    # bar_size=100 → first bar takes 40+40+20 from third trade; remainder 20 spills
    bars = aggregate_volume_bars(_trades(), bar_size=100.0)
    assert bars.height == 1  # only one full bar; remainder incomplete
    row = bars.row(0, named=True)
    assert row["volume"] == pytest.approx(100.0)
    assert row["close"] == pytest.approx(102.0)
    assert row["n_trades"] == 3


def test_volume_bar_spill_creates_second() -> None:
    trades = _trades() + [Trade(5, 104.0, 80.0, 1, 4)]
    agg = VolumeBarAggregator(bar_size=100.0)
    closed = agg.update_many(trades)
    assert len(closed) == 2
    assert all(abs(b.volume - 100.0) < 1e-9 for b in closed)


def test_aggregate_sorts_out_of_order_input() -> None:
    shuffled = list(reversed(_trades()))
    bars = aggregate_volume_bars(shuffled, bar_size=100.0)
    assert bars["start_ts"][0] <= bars["end_ts"][0]


def test_bar_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        VolumeBarAggregator(bar_size=0)

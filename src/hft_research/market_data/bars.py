"""Deterministic volume-bar construction (causal, no look-ahead)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from hft_research.exceptions import MarketDataError
from hft_research.logging_utils import get_logger
from hft_research.market_data.trades import Trade

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VolumeBar:
    """One completed volume bar."""

    bar_index: int
    start_ts_ns: int
    end_ts_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    n_trades: int
    buy_volume: float
    sell_volume: float


@dataclass
class VolumeBarAggregator:
    """Accumulate trades into fixed-volume bars deterministically.

    A bar closes when cumulative volume reaches ``bar_size``. Excess volume
    from the closing trade spills into the next bar (no look-ahead: the
    split uses only the current trade's size and price).
    """

    bar_size: float
    _bar_index: int = 0
    _cum_vol: float = 0.0
    _notional: float = 0.0
    _open: float | None = None
    _high: float = -np.inf
    _low: float = np.inf
    _close: float = 0.0
    _start_ts: int | None = None
    _end_ts: int = 0
    _n_trades: int = 0
    _buy_vol: float = 0.0
    _sell_vol: float = 0.0
    _completed: list[VolumeBar] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.bar_size <= 0:
            raise ValueError("bar_size must be positive")

    @property
    def completed(self) -> list[VolumeBar]:
        return list(self._completed)

    def reset(self) -> None:
        self._bar_index = 0
        self._reset_partial()
        self._completed.clear()

    def _reset_partial(self) -> None:
        self._cum_vol = 0.0
        self._notional = 0.0
        self._open = None
        self._high = -np.inf
        self._low = np.inf
        self._close = 0.0
        self._start_ts = None
        self._end_ts = 0
        self._n_trades = 0
        self._buy_vol = 0.0
        self._sell_vol = 0.0

    def _emit(self) -> VolumeBar:
        assert self._open is not None and self._start_ts is not None
        bar = VolumeBar(
            bar_index=self._bar_index,
            start_ts_ns=self._start_ts,
            end_ts_ns=self._end_ts,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._cum_vol,
            vwap=self._notional / self._cum_vol if self._cum_vol > 0 else self._close,
            n_trades=self._n_trades,
            buy_volume=self._buy_vol,
            sell_volume=self._sell_vol,
        )
        self._completed.append(bar)
        self._bar_index += 1
        return bar

    def _apply_piece(
        self,
        price: float,
        size: float,
        ts: int,
        side: int,
    ) -> None:
        if self._open is None:
            self._open = price
            self._start_ts = ts
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._end_ts = ts
        self._cum_vol += size
        self._notional += price * size
        self._n_trades += 1
        if side > 0:
            self._buy_vol += size
        elif side < 0:
            self._sell_vol += size

    def update(self, trade: Trade) -> list[VolumeBar]:
        """Ingest one trade; return any bars closed by this trade."""
        if trade.size <= 0:
            raise MarketDataError("Trade size must be positive")
        remaining = float(trade.size)
        price = float(trade.price)
        closed: list[VolumeBar] = []

        while remaining > 0:
            room = self.bar_size - self._cum_vol
            take = min(remaining, room)
            self._apply_piece(price, take, trade.timestamp_ns, trade.side)
            remaining -= take
            if abs(self._cum_vol - self.bar_size) < 1e-12 or self._cum_vol >= self.bar_size:
                closed.append(self._emit())
                self._reset_partial()
        return closed

    def update_many(self, trades: Sequence[Trade]) -> list[VolumeBar]:
        """Ingest trades in the given order (caller must ensure causal order)."""
        closed: list[VolumeBar] = []
        for t in trades:
            closed.extend(self.update(t))
        return closed

    def to_frame(self) -> pl.DataFrame:
        """Return completed bars as a Polars DataFrame."""
        bars = self._completed
        if not bars:
            return pl.DataFrame(
                schema={
                    "bar_index": pl.Int64,
                    "start_ts": pl.Int64,
                    "end_ts": pl.Int64,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "vwap": pl.Float64,
                    "n_trades": pl.Int64,
                    "buy_volume": pl.Float64,
                    "sell_volume": pl.Float64,
                }
            )
        return pl.DataFrame(
            {
                "bar_index": [b.bar_index for b in bars],
                "start_ts": [b.start_ts_ns for b in bars],
                "end_ts": [b.end_ts_ns for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
                "vwap": [b.vwap for b in bars],
                "n_trades": [b.n_trades for b in bars],
                "buy_volume": [b.buy_volume for b in bars],
                "sell_volume": [b.sell_volume for b in bars],
            }
        )


def aggregate_volume_bars(
    trades: Sequence[Trade],
    bar_size: float,
) -> pl.DataFrame:
    """Convenience: build volume bars from an ordered trade sequence."""
    if bar_size <= 0:
        raise ValueError("bar_size must be positive")
    # Enforce causal order without look-ahead: sort by timestamp then trade_id
    ordered = sorted(trades, key=lambda t: (t.timestamp_ns, t.trade_id or 0))
    agg = VolumeBarAggregator(bar_size=bar_size)
    agg.update_many(ordered)
    logger.debug("Formed %d volume bars from %d trades", len(agg.completed), len(ordered))
    return agg.to_frame()

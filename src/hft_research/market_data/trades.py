"""Trade tick aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from hft_research.exceptions import MarketDataError
from hft_research.market_data.timestamps import normalize_timestamps


@dataclass(frozen=True, slots=True)
class Trade:
    """Single trade print."""

    timestamp_ns: int
    price: float
    size: float
    side: int = 0  # +1 buy aggressor, -1 sell aggressor, 0 unknown
    trade_id: int | None = None


@dataclass
class TradeAggregator:
    """Causal aggregator of trade ticks into OHLCV / VWAP buckets.

    Aggregation is strictly causal: a trade is assigned to the bucket whose
    right edge is the first edge strictly greater than the trade timestamp
    when using fixed-time buckets, or appended until volume thresholds for
    volume bars (see :mod:`hft_research.market_data.bars`).
    """

    timestamp_unit: str = "ns"
    _trades: list[Trade] = field(default_factory=list, repr=False)

    def add(self, trade: Trade) -> None:
        if trade.size <= 0:
            raise MarketDataError("Trade size must be positive")
        if not np.isfinite(trade.price) or trade.price <= 0:
            raise MarketDataError("Trade price must be finite and positive")
        self._trades.append(trade)

    def add_many(self, trades: Sequence[Trade]) -> None:
        for t in trades:
            self.add(t)

    def clear(self) -> None:
        self._trades.clear()

    def __len__(self) -> int:
        return len(self._trades)

    def to_frame(self) -> pl.DataFrame:
        """Return trades as a sorted Polars DataFrame (causal order)."""
        if not self._trades:
            return pl.DataFrame(
                schema={
                    "ts": pl.Int64,
                    "price": pl.Float64,
                    "size": pl.Float64,
                    "side": pl.Int8,
                    "trade_id": pl.Int64,
                }
            )
        rows = sorted(self._trades, key=lambda t: (t.timestamp_ns, t.trade_id or 0))
        return pl.DataFrame(
            {
                "ts": [t.timestamp_ns for t in rows],
                "price": [t.price for t in rows],
                "size": [t.size for t in rows],
                "side": [t.side for t in rows],
                "trade_id": [t.trade_id if t.trade_id is not None else -1 for t in rows],
            }
        )

    @classmethod
    def from_arrays(
        cls,
        timestamps: Sequence[int | float] | np.ndarray,
        prices: Sequence[float] | np.ndarray,
        sizes: Sequence[float] | np.ndarray,
        *,
        sides: Sequence[int] | np.ndarray | None = None,
        trade_ids: Sequence[int] | np.ndarray | None = None,
        timestamp_unit: str = "ns",
    ) -> TradeAggregator:
        """Build aggregator from parallel arrays (sorted after ingest)."""
        ts = normalize_timestamps(timestamps, unit=timestamp_unit)
        px = np.asarray(prices, dtype=np.float64)
        sz = np.asarray(sizes, dtype=np.float64)
        if not (len(ts) == len(px) == len(sz)):
            raise MarketDataError("timestamp/price/size length mismatch")
        n = len(ts)
        side_arr = np.zeros(n, dtype=np.int8) if sides is None else np.asarray(sides, dtype=np.int8)
        id_arr = np.arange(n, dtype=np.int64) if trade_ids is None else np.asarray(trade_ids, dtype=np.int64)
        agg = cls(timestamp_unit=timestamp_unit)
        for i in range(n):
            agg.add(
                Trade(
                    timestamp_ns=int(ts[i]),
                    price=float(px[i]),
                    size=float(sz[i]),
                    side=int(side_arr[i]),
                    trade_id=int(id_arr[i]),
                )
            )
        return agg

    def vwap(self) -> float:
        """Volume-weighted average price of all buffered trades."""
        if not self._trades:
            raise MarketDataError("No trades to compute VWAP")
        notional = sum(t.price * t.size for t in self._trades)
        volume = sum(t.size for t in self._trades)
        if volume <= 0:
            raise MarketDataError("Total trade volume is zero")
        return notional / volume

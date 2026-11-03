"""Synthetic L2 order-book streams and trade prints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_research.config import SyntheticConfig
from hft_research.market_data.order_book import BookLevel, BookSnapshot, BookUpdate, Side
from hft_research.market_data.trades import Trade


@dataclass(frozen=True, slots=True)
class SyntheticBookStream:
    """Snapshot plus ordered incremental updates."""

    snapshot: BookSnapshot
    updates: tuple[BookUpdate, ...]
    seed: int


def _build_levels(
    mid: float,
    tick: float,
    depth: int,
    rng: np.random.Generator,
    lot: float,
) -> tuple[list[BookLevel], list[BookLevel]]:
    bids = [
        BookLevel(price=mid - tick * (i + 1), size=float(rng.integers(1, 20)) * lot)
        for i in range(depth)
    ]
    asks = [
        BookLevel(price=mid + tick * (i + 1), size=float(rng.integers(1, 20)) * lot)
        for i in range(depth)
    ]
    return bids, asks


def generate_l2_stream(
    config: SyntheticConfig | None = None,
    *,
    seed: int | None = None,
    n_events: int | None = None,
    depth: int = 5,
    start_mid: float = 100.0,
) -> SyntheticBookStream:
    """Generate a consistent L2 snapshot + incremental updates.

    Updates never cross the book. Sequence numbers are contiguous from the
    snapshot sequence. Suitable for reconstruction / gap tests.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 7)
    n = int(n_events if n_events is not None else cfg.n_book_events)
    tick = cfg.tick_size
    lot = cfg.lot_size

    bids, asks = _build_levels(start_mid, tick, depth, rng, lot)
    snapshot = BookSnapshot(
        sequence=0,
        timestamp_ns=0,
        bids=tuple(bids),
        asks=tuple(asks),
    )

    # Maintain mutable top-of-book style state for consistency
    bid_map = {b.price: b.size for b in bids}
    ask_map = {a.price: a.size for a in asks}
    updates: list[BookUpdate] = []
    mid = start_mid
    t_ns = 0

    for seq in range(1, n + 1):
        t_ns += int(rng.integers(1_000, 50_000))  # 1–50 µs
        # Random walk mid slightly; rebuild if needed to avoid crosses
        mid += float(rng.normal(0.0, tick * 0.15))
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        level_idx = int(rng.integers(0, depth))
        if side is Side.BID:
            price = round(mid - tick * (level_idx + 1), 10)
            # Ensure strictly below best ask
            best_ask = min(ask_map) if ask_map else mid + tick
            if price >= best_ask:
                price = best_ask - tick
            size = float(rng.integers(0, 25)) * lot  # 0 = delete
            if size <= 0:
                bid_map.pop(price, None)
            else:
                bid_map[price] = size
        else:
            price = round(mid + tick * (level_idx + 1), 10)
            best_bid = max(bid_map) if bid_map else mid - tick
            if price <= best_bid:
                price = best_bid + tick
            size = float(rng.integers(0, 25)) * lot
            if size <= 0:
                ask_map.pop(price, None)
            else:
                ask_map[price] = size

        # Ensure both sides non-empty
        if not bid_map:
            px = round(mid - tick, 10)
            bid_map[px] = lot
            price, size, side = px, lot, Side.BID
        if not ask_map:
            px = round(mid + tick, 10)
            ask_map[px] = lot
            price, size, side = px, lot, Side.ASK

        updates.append(
            BookUpdate(
                sequence=seq,
                timestamp_ns=t_ns,
                side=side,
                price=float(price),
                size=float(size),
            )
        )

    return SyntheticBookStream(
        snapshot=snapshot,
        updates=tuple(updates),
        seed=int(seed if seed is not None else cfg.seed + 7),
    )


def generate_trades(
    config: SyntheticConfig | None = None,
    *,
    seed: int | None = None,
    n: int = 5000,
    start_price: float = 100.0,
) -> list[Trade]:
    """Generate a causal trade tape with positive sizes."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 99)
    trades: list[Trade] = []
    px = start_price
    t_ns = 0
    for i in range(n):
        t_ns += int(rng.integers(10_000, 200_000))
        px = max(cfg.tick_size, px + float(rng.normal(0.0, cfg.tick_size)))
        px = round(px / cfg.tick_size) * cfg.tick_size
        size = float(rng.integers(1, 50)) * cfg.lot_size
        side = 1 if rng.random() < 0.5 else -1
        trades.append(
            Trade(
                timestamp_ns=t_ns,
                price=float(px),
                size=size,
                side=side,
                trade_id=i,
            )
        )
    return trades

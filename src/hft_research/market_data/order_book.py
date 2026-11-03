"""L2 order book reconstruction with snapshot / incremental updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from hft_research.exceptions import BookIntegrityError, SequenceGapError
from hft_research.logging_utils import get_logger

logger = get_logger(__name__)


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


@dataclass(frozen=True, slots=True, order=True)
class BookLevel:
    """Single price level."""

    price: float
    size: float


@dataclass(frozen=True, slots=True)
class BookUpdate:
    """Incremental L2 update for a single side/price."""

    sequence: int
    timestamp_ns: int
    side: Side
    price: float
    size: float  # 0 means delete the level


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Full L2 snapshot used for bootstrap / recovery."""

    sequence: int
    timestamp_ns: int
    bids: tuple[BookLevel, ...]  # descending price
    asks: tuple[BookLevel, ...]  # ascending price


@dataclass
class OrderBook:
    """Deterministic L2 order book with sequence validation.

    Bids are stored price-descending; asks price-ascending. Size ``0`` on an
    incremental update deletes the level. Sequence gaps larger than
    ``max_sequence_gap`` raise :class:`SequenceGapError` and mark the book
    as needing a snapshot for recovery.
    """

    depth: int = 10
    max_sequence_gap: int = 1
    _bids: dict[float, float] = field(default_factory=dict, repr=False)
    _asks: dict[float, float] = field(default_factory=dict, repr=False)
    _last_sequence: int | None = None
    _last_timestamp_ns: int | None = None
    _needs_snapshot: bool = True
    _update_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def last_sequence(self) -> int | None:
        return self._last_sequence

    @property
    def last_timestamp_ns(self) -> int | None:
        return self._last_timestamp_ns

    @property
    def needs_snapshot(self) -> bool:
        return self._needs_snapshot

    @property
    def update_count(self) -> int:
        return self._update_count

    def clear(self) -> None:
        """Reset book state; requires a fresh snapshot before use."""
        self._bids.clear()
        self._asks.clear()
        self._last_sequence = None
        self._last_timestamp_ns = None
        self._needs_snapshot = True
        self._update_count = 0

    # ------------------------------------------------------------------
    # Snapshot / incremental
    # ------------------------------------------------------------------
    def apply_snapshot(self, snapshot: BookSnapshot) -> None:
        """Replace book state with a full snapshot (recovery path)."""
        if snapshot.sequence < 0:
            raise BookIntegrityError("Snapshot sequence must be non-negative")
        self._bids = {lvl.price: lvl.size for lvl in snapshot.bids if lvl.size > 0}
        self._asks = {lvl.price: lvl.size for lvl in snapshot.asks if lvl.size > 0}
        self._last_sequence = snapshot.sequence
        self._last_timestamp_ns = snapshot.timestamp_ns
        self._needs_snapshot = False
        self._update_count += 1
        self._trim_to_depth()
        self._validate_integrity()
        logger.debug(
            "Applied snapshot seq=%s bids=%d asks=%d",
            snapshot.sequence,
            len(self._bids),
            len(self._asks),
        )

    def apply_update(self, update: BookUpdate) -> None:
        """Apply an incremental update with sequence validation."""
        if self._needs_snapshot or self._last_sequence is None:
            raise SequenceGapError(
                expected=-1,
                received=update.sequence,
                message="Book requires snapshot before incremental updates",
            )
        expected = self._last_sequence + 1
        gap = update.sequence - expected
        if gap < 0:
            # Duplicate / stale — ignore but log
            logger.debug(
                "Ignoring stale/duplicate update seq=%s last=%s",
                update.sequence,
                self._last_sequence,
            )
            return
        if gap > self.max_sequence_gap:
            self._needs_snapshot = True
            raise SequenceGapError(expected=expected, received=update.sequence)

        book = self._bids if update.side is Side.BID else self._asks
        if update.size <= 0:
            book.pop(update.price, None)
        else:
            book[update.price] = float(update.size)

        self._last_sequence = update.sequence
        self._last_timestamp_ns = update.timestamp_ns
        self._update_count += 1
        self._trim_to_depth()
        self._validate_integrity()

    def apply_updates(self, updates: Sequence[BookUpdate]) -> None:
        """Apply a batch of updates in sequence order."""
        for upd in sorted(updates, key=lambda u: u.sequence):
            self.apply_update(upd)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def bids(self, n: int | None = None) -> list[BookLevel]:
        """Return top-n bids (best first)."""
        depth = n if n is not None else self.depth
        prices = sorted(self._bids.keys(), reverse=True)[:depth]
        return [BookLevel(p, self._bids[p]) for p in prices]

    def asks(self, n: int | None = None) -> list[BookLevel]:
        """Return top-n asks (best first)."""
        depth = n if n is not None else self.depth
        prices = sorted(self._asks.keys())[:depth]
        return [BookLevel(p, self._asks[p]) for p in prices]

    def best_bid(self) -> BookLevel | None:
        levels = self.bids(1)
        return levels[0] if levels else None

    def best_ask(self) -> BookLevel | None:
        levels = self.asks(1)
        return levels[0] if levels else None

    def mid_price(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            raise BookIntegrityError("Cannot compute mid: empty side")
        return 0.5 * (bb.price + ba.price)

    def spread(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            raise BookIntegrityError("Cannot compute spread: empty side")
        return ba.price - bb.price

    def to_arrays(self, n: int | None = None) -> Mapping[str, np.ndarray]:
        """Export top-n levels as NumPy arrays for analytics."""
        b = self.bids(n)
        a = self.asks(n)
        return {
            "bid_px": np.array([x.price for x in b], dtype=np.float64),
            "bid_sz": np.array([x.size for x in b], dtype=np.float64),
            "ask_px": np.array([x.price for x in a], dtype=np.float64),
            "ask_sz": np.array([x.size for x in a], dtype=np.float64),
        }

    def snapshot(self) -> BookSnapshot:
        """Emit current state as a snapshot (for persistence / recovery)."""
        if self._last_sequence is None or self._last_timestamp_ns is None:
            raise BookIntegrityError("Cannot snapshot uninitialized book")
        return BookSnapshot(
            sequence=self._last_sequence,
            timestamp_ns=self._last_timestamp_ns,
            bids=tuple(self.bids()),
            asks=tuple(self.asks()),
        )

    # ------------------------------------------------------------------
    # Integrity / maintenance
    # ------------------------------------------------------------------
    def _trim_to_depth(self) -> None:
        """Retain only the best ``depth`` levels on each side."""
        if len(self._bids) > self.depth:
            keep = sorted(self._bids.keys(), reverse=True)[: self.depth]
            self._bids = {p: self._bids[p] for p in keep}
        if len(self._asks) > self.depth:
            keep = sorted(self._asks.keys())[: self.depth]
            self._asks = {p: self._asks[p] for p in keep}

    def _validate_integrity(self) -> None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None and bb.price >= ba.price:
            raise BookIntegrityError(
                f"Crossed book: best_bid={bb.price} >= best_ask={ba.price}"
            )
        for side_name, book in (("bid", self._bids), ("ask", self._asks)):
            for px, sz in book.items():
                if not np.isfinite(px) or not np.isfinite(sz):
                    raise BookIntegrityError(f"Non-finite {side_name} level")
                if sz < 0:
                    raise BookIntegrityError(f"Negative size on {side_name} @ {px}")


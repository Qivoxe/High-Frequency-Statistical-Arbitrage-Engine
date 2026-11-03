"""Order book reconstruction tests."""

from __future__ import annotations

import pytest

from hft_research.exceptions import BookIntegrityError, SequenceGapError
from hft_research.market_data.order_book import (
    BookLevel,
    BookSnapshot,
    BookUpdate,
    OrderBook,
    Side,
)
from hft_research.synthetic.order_book_synth import generate_l2_stream


def _snapshot(seq: int = 0) -> BookSnapshot:
    return BookSnapshot(
        sequence=seq,
        timestamp_ns=0,
        bids=(BookLevel(99.0, 10.0), BookLevel(98.0, 5.0)),
        asks=(BookLevel(101.0, 8.0), BookLevel(102.0, 4.0)),
    )


def test_snapshot_bootstrap() -> None:
    book = OrderBook(depth=2)
    book.apply_snapshot(_snapshot())
    assert book.best_bid() == BookLevel(99.0, 10.0)
    assert book.best_ask() == BookLevel(101.0, 8.0)
    assert book.mid_price() == 100.0
    assert book.spread() == 2.0
    assert not book.needs_snapshot


def test_incremental_update_and_delete() -> None:
    book = OrderBook()
    book.apply_snapshot(_snapshot())
    book.apply_update(BookUpdate(1, 10, Side.BID, 99.0, 20.0))
    assert book.best_bid().size == 20.0
    book.apply_update(BookUpdate(2, 20, Side.BID, 99.0, 0.0))
    assert book.best_bid().price == 98.0


def test_requires_snapshot_before_update() -> None:
    book = OrderBook()
    with pytest.raises(SequenceGapError):
        book.apply_update(BookUpdate(1, 0, Side.BID, 99.0, 1.0))


def test_sequence_gap_triggers_recovery_flag() -> None:
    book = OrderBook(max_sequence_gap=1)
    book.apply_snapshot(_snapshot())
    with pytest.raises(SequenceGapError) as exc:
        book.apply_update(BookUpdate(5, 10, Side.ASK, 101.0, 1.0))
    assert book.needs_snapshot
    assert exc.value.expected == 1
    assert exc.value.received == 5


def test_stale_update_ignored() -> None:
    book = OrderBook()
    book.apply_snapshot(_snapshot())
    book.apply_update(BookUpdate(1, 10, Side.ASK, 101.0, 9.0))
    book.apply_update(BookUpdate(1, 11, Side.ASK, 101.0, 100.0))  # stale
    assert book.best_ask().size == 9.0
    assert book.last_sequence == 1


def test_crossed_book_rejected() -> None:
    book = OrderBook()
    book.apply_snapshot(_snapshot())
    with pytest.raises(BookIntegrityError):
        book.apply_update(BookUpdate(1, 10, Side.BID, 102.0, 1.0))


def test_synthetic_stream_reconstructs(book_stream) -> None:
    book = OrderBook(depth=5, max_sequence_gap=1)
    book.apply_snapshot(book_stream.snapshot)
    for upd in book_stream.updates:
        book.apply_update(upd)
    assert book.last_sequence == book_stream.updates[-1].sequence
    assert book.best_bid() is not None
    assert book.best_ask() is not None
    assert book.best_bid().price < book.best_ask().price


def test_snapshot_roundtrip() -> None:
    stream = generate_l2_stream(n_events=50, depth=3, seed=1)
    book = OrderBook(depth=3)
    book.apply_snapshot(stream.snapshot)
    for upd in stream.updates[:20]:
        book.apply_update(upd)
    snap = book.snapshot()
    book2 = OrderBook(depth=3)
    book2.apply_snapshot(snap)
    assert book2.best_bid() == book.best_bid()
    assert book2.best_ask() == book.best_ask()

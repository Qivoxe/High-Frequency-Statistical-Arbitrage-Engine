"""Micro-price and imbalance unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.exceptions import BookIntegrityError
from hft_research.market_data.imbalance import order_book_imbalance
from hft_research.market_data.microprice import micro_price
from hft_research.market_data.order_book import BookLevel, BookSnapshot, OrderBook


def _book(bid_sz: float = 10.0, ask_sz: float = 10.0) -> OrderBook:
    snap = BookSnapshot(
        sequence=0,
        timestamp_ns=0,
        bids=(BookLevel(100.0, bid_sz), BookLevel(99.0, 5.0)),
        asks=(BookLevel(101.0, ask_sz), BookLevel(102.0, 5.0)),
    )
    book = OrderBook(depth=2)
    book.apply_snapshot(snap)
    return book


def test_microprice_top_of_book_symmetric() -> None:
    # Equal sizes → mid
    mp = micro_price(
        bid_px=np.array([100.0]),
        bid_sz=np.array([10.0]),
        ask_px=np.array([101.0]),
        ask_sz=np.array([10.0]),
        top_n=1,
    )
    assert mp == pytest.approx(100.5)


def test_microprice_tilts_toward_scarce_side() -> None:
    # Large bid size → micro closer to ask (pressure to lift)
    mp = micro_price(
        bid_px=np.array([100.0]),
        bid_sz=np.array([90.0]),
        ask_px=np.array([101.0]),
        ask_sz=np.array([10.0]),
        top_n=1,
    )
    assert mp > 100.5


def test_microprice_from_book() -> None:
    book = _book(10.0, 10.0)
    assert micro_price(book, top_n=1) == pytest.approx(100.5)


def test_imbalance_bounds() -> None:
    assert order_book_imbalance(bid_sz=np.array([10.0]), ask_sz=np.array([10.0])) == 0.0
    assert order_book_imbalance(bid_sz=np.array([10.0]), ask_sz=np.array([0.0])) == 1.0
    assert order_book_imbalance(bid_sz=np.array([0.0]), ask_sz=np.array([10.0])) == -1.0


def test_imbalance_from_book() -> None:
    book = _book(30.0, 10.0)
    imb = order_book_imbalance(book, top_n=1)
    assert imb == pytest.approx(0.5)


def test_zero_volume_raises() -> None:
    with pytest.raises(BookIntegrityError):
        order_book_imbalance(bid_sz=np.array([0.0]), ask_sz=np.array([0.0]))
    with pytest.raises(BookIntegrityError):
        micro_price(
            bid_px=np.array([1.0]),
            bid_sz=np.array([0.0]),
            ask_px=np.array([2.0]),
            ask_sz=np.array([1.0]),
        )

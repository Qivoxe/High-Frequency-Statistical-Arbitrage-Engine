"""Micro-price calculation from L2 order book levels."""

from __future__ import annotations

import numpy as np

from hft_research.exceptions import BookIntegrityError
from hft_research.market_data.order_book import OrderBook


def micro_price(
    book: OrderBook | None = None,
    *,
    bid_px: np.ndarray | None = None,
    bid_sz: np.ndarray | None = None,
    ask_px: np.ndarray | None = None,
    ask_sz: np.ndarray | None = None,
    top_n: int = 1,
) -> float:
    """Compute size-weighted micro-price over the top ``top_n`` levels.

    For top-of-book (``top_n=1``)::

        micro = (ask_px * bid_sz + bid_px * ask_sz) / (bid_sz + ask_sz)

    For multi-level, sizes are summed and prices are size-weighted within
    each side before applying the same formula. This is the standard
    Cont–Kukanov–Stoikov style micro-price used in HFT research.

    Parameters
    ----------
    book:
        Optional live :class:`OrderBook`. If provided, arrays are extracted.
    bid_px, bid_sz, ask_px, ask_sz:
        Level arrays (best first). Required if ``book`` is ``None``.
    top_n:
        Number of levels to aggregate on each side (>=1).

    Returns
    -------
    float
        Micro-price.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    if book is not None:
        arrays = book.to_arrays(top_n)
        bid_px = arrays["bid_px"]
        bid_sz = arrays["bid_sz"]
        ask_px = arrays["ask_px"]
        ask_sz = arrays["ask_sz"]
    else:
        if bid_px is None or bid_sz is None or ask_px is None or ask_sz is None:
            raise ValueError("Provide either book or all of bid/ask px/sz arrays")
        bid_px = np.asarray(bid_px, dtype=np.float64)[:top_n]
        bid_sz = np.asarray(bid_sz, dtype=np.float64)[:top_n]
        ask_px = np.asarray(ask_px, dtype=np.float64)[:top_n]
        ask_sz = np.asarray(ask_sz, dtype=np.float64)[:top_n]

    if bid_px.size == 0 or ask_px.size == 0:
        raise BookIntegrityError("Cannot compute micro-price: empty side")

    bid_vol = float(np.sum(bid_sz))
    ask_vol = float(np.sum(ask_sz))
    if bid_vol <= 0 or ask_vol <= 0:
        raise BookIntegrityError("Cannot compute micro-price: zero size on a side")

    # Size-weighted price per side
    bid_vwap = float(np.dot(bid_px, bid_sz) / bid_vol)
    ask_vwap = float(np.dot(ask_px, ask_sz) / ask_vol)

    return (ask_vwap * bid_vol + bid_vwap * ask_vol) / (bid_vol + ask_vol)

"""Order book imbalance metrics."""

from __future__ import annotations

import numpy as np

from hft_research.exceptions import BookIntegrityError
from hft_research.market_data.order_book import OrderBook


def order_book_imbalance(
    book: OrderBook | None = None,
    *,
    bid_sz: np.ndarray | None = None,
    ask_sz: np.ndarray | None = None,
    top_n: int = 1,
) -> float:
    """Compute signed order-book imbalance in ``[-1, 1]``.

    Definition::

        I = (V_bid - V_ask) / (V_bid + V_ask)

    where volumes are summed over the top ``top_n`` levels.

    Returns
    -------
    float
        Imbalance; ``+1`` = all bid volume, ``-1`` = all ask volume.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    if book is not None:
        arrays = book.to_arrays(top_n)
        bid_sz = arrays["bid_sz"]
        ask_sz = arrays["ask_sz"]
    else:
        if bid_sz is None or ask_sz is None:
            raise ValueError("Provide either book or bid_sz and ask_sz")
        bid_sz = np.asarray(bid_sz, dtype=np.float64)[:top_n]
        ask_sz = np.asarray(ask_sz, dtype=np.float64)[:top_n]

    v_bid = float(np.sum(bid_sz))
    v_ask = float(np.sum(ask_sz))
    denom = v_bid + v_ask
    if denom <= 0:
        raise BookIntegrityError("Cannot compute imbalance: zero total volume")
    return (v_bid - v_ask) / denom

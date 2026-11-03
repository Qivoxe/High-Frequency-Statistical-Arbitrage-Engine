"""Timestamp normalization and causal ordering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from hft_research.exceptions import TimestampError
from hft_research.logging_utils import get_logger

logger = get_logger(__name__)

_UNIT_TO_NS: dict[str, int] = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
}


def unit_to_ns_factor(unit: str) -> int:
    """Return multiplicative factor to convert ``unit`` to nanoseconds."""
    key = unit.lower().strip()
    if key not in _UNIT_TO_NS:
        raise TimestampError(f"Unsupported timestamp unit: {unit!r}")
    return _UNIT_TO_NS[key]


def normalize_timestamps(
    values: Sequence[int | float] | np.ndarray,
    *,
    unit: str = "ns",
) -> np.ndarray:
    """Convert timestamps to int64 UTC nanoseconds.

    Parameters
    ----------
    values:
        Raw timestamps in the given unit.
    unit:
        One of ``ns``, ``us``, ``ms``, ``s``.

    Returns
    -------
    np.ndarray
        Int64 array of nanosecond timestamps.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    if np.any(~np.isfinite(arr)):
        raise TimestampError("Timestamps contain NaN or Inf")
    if np.any(arr < 0):
        raise TimestampError("Negative timestamps are not allowed")
    factor = unit_to_ns_factor(unit)
    out = np.rint(arr * factor).astype(np.int64)
    return out


@dataclass
class TimestampNormalizer:
    """Stateful normalizer that rejects causal violations optionally."""

    unit: str = "ns"
    max_out_of_order_ns: int = 0
    _last_accepted: int | None = None
    _dropped: int = 0

    def reset(self) -> None:
        self._last_accepted = None
        self._dropped = 0

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def normalize_one(self, value: int | float) -> int | None:
        """Normalize a single timestamp; drop if too far out of order.

        Returns
        -------
        int | None
            Accepted nanosecond timestamp, or ``None`` if dropped.
        """
        ts = int(normalize_timestamps([value], unit=self.unit)[0])
        if self._last_accepted is None:
            self._last_accepted = ts
            return ts
        if ts < self._last_accepted:
            lag = self._last_accepted - ts
            if self.max_out_of_order_ns > 0 and lag > self.max_out_of_order_ns:
                self._dropped += 1
                logger.debug("Dropping out-of-order ts=%s lag_ns=%s", ts, lag)
                return None
            # Allow mild reordering but do not move the watermark backward.
            return ts
        self._last_accepted = ts
        return ts

    def normalize_series(self, values: Iterable[int | float]) -> np.ndarray:
        """Normalize a series, dropping causal violators when configured."""
        accepted: list[int] = []
        for v in values:
            ts = self.normalize_one(v)
            if ts is not None:
                accepted.append(ts)
        return np.asarray(accepted, dtype=np.int64)

    def sort_frame(
        self,
        df: pl.DataFrame,
        time_col: str = "ts",
        *,
        unit: str | None = None,
    ) -> pl.DataFrame:
        """Normalize and stably sort a Polars frame by time (causal order).

        Does not forward-fill; only sorts and normalizes. Prevents look-ahead
        by ensuring consumers iterate in ascending time order.
        """
        if time_col not in df.columns:
            raise TimestampError(f"Missing time column: {time_col}")
        use_unit = unit or self.unit
        raw = df[time_col].to_numpy()
        ns = normalize_timestamps(raw, unit=use_unit)
        out = df.with_columns(pl.Series(time_col, ns)).sort(time_col, maintain_order=True)
        return out

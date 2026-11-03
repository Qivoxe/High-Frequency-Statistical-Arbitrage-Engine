"""Reproducible synthetic datasets for Phase 1 validation."""

from hft_research.synthetic.cointegrated import (
    SyntheticPairResult,
    generate_cointegrated_pair,
    generate_independent_pair,
    generate_universe,
)
from hft_research.synthetic.order_book_synth import (
    SyntheticBookStream,
    generate_l2_stream,
    generate_trades,
)

__all__ = [
    "SyntheticPairResult",
    "generate_cointegrated_pair",
    "generate_independent_pair",
    "generate_universe",
    "SyntheticBookStream",
    "generate_l2_stream",
    "generate_trades",
]

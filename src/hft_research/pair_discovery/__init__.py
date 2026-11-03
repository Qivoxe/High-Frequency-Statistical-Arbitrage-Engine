"""Pair discovery: cointegration testing, stability, ranking."""

from hft_research.pair_discovery.engle_granger import engle_granger_test
from hft_research.pair_discovery.johansen import johansen_test
from hft_research.pair_discovery.multiple_testing import correct_pvalues
from hft_research.pair_discovery.ranking import PairCandidate, rank_pairs
from hft_research.pair_discovery.stability import rolling_cointegration_stability
from hft_research.pair_discovery.stationarity import adf_test, half_life_ou

__all__ = [
    "adf_test",
    "half_life_ou",
    "engle_granger_test",
    "johansen_test",
    "correct_pvalues",
    "rolling_cointegration_stability",
    "PairCandidate",
    "rank_pairs",
]

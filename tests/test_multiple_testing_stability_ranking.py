"""Multiple testing, rolling stability, and pair ranking tests."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.config import PairDiscoveryConfig
from hft_research.exceptions import PairDiscoveryError
from hft_research.pair_discovery.multiple_testing import correct_pvalues
from hft_research.pair_discovery.ranking import rank_pairs
from hft_research.pair_discovery.stability import rolling_cointegration_stability
from hft_research.synthetic.cointegrated import generate_cointegrated_pair, generate_universe


def test_bonferroni_and_bh() -> None:
    p = np.array([0.001, 0.04, 0.2, 0.5])
    bonf = correct_pvalues(p, method="bonferroni", alpha=0.05)
    assert bonf.corrected_pvalues[0] == pytest.approx(0.004)
    assert bonf.rejected[0]
    assert not bonf.rejected[2]

    bh = correct_pvalues(p, method="benjamini_hochberg", alpha=0.05)
    assert bh.rejected[0]
    assert bh.corrected_pvalues[0] <= bonf.corrected_pvalues[0] + 1e-12


def test_invalid_pvalues() -> None:
    with pytest.raises(PairDiscoveryError):
        correct_pvalues(np.array([-0.1, 0.2]))


def test_rolling_stability_high_for_coint(cointegrated_pair) -> None:
    res = rolling_cointegration_stability(
        cointegrated_pair.y,
        cointegrated_pair.x,
        window=150,
        step=25,
        significance=0.05,
    )
    assert res.n_windows >= 2
    assert res.stability_ratio >= 0.5
    assert res.hedge_ratio_cv < 0.5


def test_rank_pairs_finds_planted() -> None:
    uni = generate_universe(
        seed=42,
        n_cointegrated_pairs=1,
        n_noise_assets=2,
    )
    # Ensure planted pair length
    cfg = PairDiscoveryConfig(
        min_obs=100,
        rolling_window=150,
        rolling_step=30,
        min_stability_ratio=0.3,
        half_life_min=1.0,
        half_life_max=100.0,
        fdr_alpha=0.10,
    )
    # Truncate to manageable size from generator default
    uni = {k: v[:600] for k, v in uni.items()}
    ranked = rank_pairs(uni, config=cfg)
    assert ranked
    top_names = {(c.asset_y, c.asset_x) for c in ranked[:3]}
    assert ("Y0", "X0") in top_names or ("X0", "Y0") in {
        (c.asset_x, c.asset_y) for c in ranked[:3]
    } or any({c.asset_y, c.asset_x} == {"Y0", "X0"} for c in ranked[:5])


def test_rank_pairs_scores_coint_above_noise() -> None:
    pair = generate_cointegrated_pair(seed=7, n=600)
    rng = np.random.default_rng(7)
    noise = 100 + np.cumsum(rng.normal(0, 0.01, size=600))
    prices = {"Y": pair.y, "X": pair.x, "N": noise}
    cfg = PairDiscoveryConfig(
        min_obs=100,
        rolling_window=120,
        rolling_step=20,
        min_stability_ratio=0.2,
        half_life_max=120.0,
    )
    ranked = rank_pairs(prices, config=cfg)
    yx = next(c for c in ranked if {c.asset_y, c.asset_x} == {"Y", "X"})
    yn = next(c for c in ranked if {c.asset_y, c.asset_x} == {"Y", "N"})
    assert yx.score >= yn.score

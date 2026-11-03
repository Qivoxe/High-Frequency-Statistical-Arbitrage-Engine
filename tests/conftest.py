"""Shared pytest fixtures for Phase 1."""

from __future__ import annotations

import numpy as np
import pytest

from hft_research.config import (
    HedgingConfig,
    MarketDataConfig,
    PairDiscoveryConfig,
    Phase1Config,
    SyntheticConfig,
)
from hft_research.synthetic.cointegrated import (
    generate_cointegrated_pair,
    generate_independent_pair,
)
from hft_research.synthetic.order_book_synth import generate_l2_stream, generate_trades


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_cfg() -> SyntheticConfig:
    return SyntheticConfig(seed=42, n_bars=800, n_book_events=500)


@pytest.fixture
def cointegrated_pair(synthetic_cfg: SyntheticConfig):
    return generate_cointegrated_pair(synthetic_cfg)


@pytest.fixture
def independent_pair(synthetic_cfg: SyntheticConfig):
    return generate_independent_pair(synthetic_cfg)


@pytest.fixture
def book_stream(synthetic_cfg: SyntheticConfig):
    return generate_l2_stream(synthetic_cfg, depth=5, n_events=200)


@pytest.fixture
def trades(synthetic_cfg: SyntheticConfig):
    return generate_trades(synthetic_cfg, n=1500)


@pytest.fixture
def phase1_cfg(synthetic_cfg: SyntheticConfig) -> Phase1Config:
    return Phase1Config(
        market_data=MarketDataConfig(book_depth=5, volume_bar_size=100.0, top_n_levels=3),
        pair_discovery=PairDiscoveryConfig(
            min_obs=80,
            rolling_window=120,
            rolling_step=20,
            min_stability_ratio=0.40,
            half_life_min=1.0,
            half_life_max=80.0,
        ),
        hedging=HedgingConfig(
            adaptive_process_noise=True,
            observation_noise=1e-3,
            process_noise_diag=(1e-5, 1e-6),
        ),
        synthetic=synthetic_cfg,
    )

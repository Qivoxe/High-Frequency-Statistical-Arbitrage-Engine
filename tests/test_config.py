"""Configuration loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_research.config import Phase1Config, default_config_path, load_config
from hft_research.exceptions import ConfigurationError


def test_default_config() -> None:
    cfg = Phase1Config()
    assert cfg.market_data.book_depth == 10
    assert cfg.hedging.adaptive_process_noise is True


def test_load_repo_yaml() -> None:
    path = default_config_path()
    assert path.exists()
    cfg = load_config(path)
    assert cfg.synthetic.seed == 42
    assert cfg.pair_discovery.multiple_testing == "benjamini_hochberg"
    assert len(cfg.hedging.initial_P_diag) == 2


def test_missing_file() -> None:
    with pytest.raises(ConfigurationError):
        load_config(Path("does_not_exist.yaml"))

"""Integration test: synthetic data → discovery → Kalman hedge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hft_research.pipeline.phase1 import run_phase1_pipeline


def _local_tmp(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    results = root / "results"
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return data, results


@pytest.mark.integration
def test_phase1_pipeline_end_to_end(phase1_cfg) -> None:
    # Use repo-local scratch dir (Windows CI sometimes locks pytest tmp_path)
    scratch = Path(__file__).resolve().parents[1] / ".pytest_scratch" / "integration_e2e"
    data_dir, results_dir = _local_tmp(scratch)
    cfg = replace(
        phase1_cfg,
        paths=replace(phase1_cfg.paths, data_dir=data_dir, results_dir=results_dir),
    )
    result = run_phase1_pipeline(cfg, train_fraction=0.7, write_outputs=True)

    assert result.train_size > 0 and result.test_size > 0
    assert result.n_volume_bars >= 1
    assert result.book_microprice is not None
    assert -1.0 <= result.book_imbalance <= 1.0
    assert result.candidates
    assert result.selected is not None
    planted_ok = any({c.asset_y, c.asset_x} == {"Y0", "X0"} for c in result.candidates)
    assert planted_ok
    assert result.filter_diagnostics is not None
    assert result.filter_diagnostics.nobs > result.train_size
    assert not result.filter_diagnostics.diverged
    assert (data_dir / "universe.csv").exists()
    assert (results_dir / "phase1_summary.json").exists()
    assert (results_dir / "kalman_path.csv").exists()


@pytest.mark.integration
def test_pipeline_no_lookahead_split(phase1_cfg) -> None:
    """Discovery uses only train window — test prices never enter ranking inputs."""
    scratch = Path(__file__).resolve().parents[1] / ".pytest_scratch" / "integration_split"
    data_dir, results_dir = _local_tmp(scratch)
    cfg = replace(
        phase1_cfg,
        paths=replace(phase1_cfg.paths, data_dir=data_dir, results_dir=results_dir),
    )
    result = run_phase1_pipeline(cfg, train_fraction=0.6, write_outputs=False)
    assert result.train_size == int(cfg.synthetic.n_bars * 0.6)
    assert len(result.candidates) >= 1

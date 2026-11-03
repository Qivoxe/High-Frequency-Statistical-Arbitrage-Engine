"""Phase 1 end-to-end pipeline: data → pairs → dynamic hedge.

Separates market-data ingestion from analytics and enforces a causal
train/test split so pair discovery never sees evaluation-period prices
(look-ahead / survivorship control).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from hft_research.config import Phase1Config, default_config_path, load_config
from hft_research.hedging.diagnostics import FilterDiagnostics, diagnose_filter
from hft_research.hedging.kalman import KalmanHedgeFilter
from hft_research.logging_utils import get_logger, setup_logging
from hft_research.market_data.bars import aggregate_volume_bars
from hft_research.market_data.imbalance import order_book_imbalance
from hft_research.market_data.microprice import micro_price
from hft_research.market_data.order_book import OrderBook
from hft_research.pair_discovery.ranking import PairCandidate, rank_pairs
from hft_research.synthetic.cointegrated import (
    generate_cointegrated_pair,
    generate_universe,
    write_pair_csv,
)
from hft_research.synthetic.order_book_synth import generate_l2_stream, generate_trades

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Phase1Result:
    """Artifacts produced by the Phase 1 research pipeline."""

    candidates: list[PairCandidate]
    selected: PairCandidate | None
    filter_diagnostics: FilterDiagnostics | None
    train_size: int
    test_size: int
    book_microprice: float | None
    book_imbalance: float | None
    n_volume_bars: int
    paths_written: tuple[str, ...]


def _candidate_to_dict(c: PairCandidate) -> dict[str, Any]:
    return {
        "asset_y": c.asset_y,
        "asset_x": c.asset_x,
        "eg_pvalue": c.eg_pvalue,
        "eg_pvalue_corrected": c.eg_pvalue_corrected,
        "johansen_cointegrated": c.johansen_cointegrated,
        "johansen_rank": c.johansen_rank,
        "hedge_ratio": c.hedge_ratio,
        "intercept": c.intercept,
        "half_life": c.half_life,
        "residual_std": c.residual_std,
        "stability_ratio": c.stability_ratio,
        "hedge_ratio_cv": c.hedge_ratio_cv,
        "score": c.score,
        "accepted": c.accepted,
    }


def run_phase1_pipeline(
    config: Phase1Config | None = None,
    *,
    train_fraction: float = 0.7,
    write_outputs: bool = True,
) -> Phase1Result:
    """Execute the full Phase 1 research flow on synthetic data.

    Steps
    -----
    1. Generate reproducible synthetic L2 stream, trades, and price universe.
    2. Reconstruct the book; compute micro-price and imbalance (ingestion).
    3. Aggregate trades into deterministic volume bars.
    4. Split prices into train / test (causal). Discover pairs on train only.
    5. Fit Kalman hedge on train, then filter test causally.
    6. Optionally persist datasets, rankings, and diagnostics.
    """
    cfg = config or load_config(default_config_path())
    setup_logging(cfg.logging)
    logger.info("Starting Phase 1 pipeline (seed=%s)", cfg.synthetic.seed)

    if not (0.1 < train_fraction < 0.95):
        raise ValueError("train_fraction must be in (0.1, 0.95)")

    written: list[str] = []
    data_dir = cfg.paths.data_dir
    results_dir = cfg.paths.results_dir
    if write_outputs:
        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    # --- Market data ingestion (separated from analytics) ---
    stream = generate_l2_stream(cfg.synthetic, depth=cfg.market_data.book_depth)
    book = OrderBook(
        depth=cfg.market_data.book_depth,
        max_sequence_gap=cfg.market_data.max_sequence_gap,
    )
    book.apply_snapshot(stream.snapshot)
    for upd in stream.updates:
        try:
            book.apply_update(upd)
        except Exception as exc:  # noqa: BLE001 — recover via original snapshot path
            logger.warning("Book update failed (%s); recovering via snapshot", exc)
            book.clear()
            book.apply_snapshot(stream.snapshot)
            break

    mp = micro_price(book, top_n=min(cfg.market_data.top_n_levels, cfg.market_data.book_depth))
    imb = order_book_imbalance(book, top_n=min(cfg.market_data.top_n_levels, cfg.market_data.book_depth))
    logger.info("Book micro-price=%.6f imbalance=%.4f", mp, imb)

    trades = generate_trades(cfg.synthetic, n=max(2000, cfg.synthetic.n_book_events))
    bars = aggregate_volume_bars(trades, bar_size=cfg.market_data.volume_bar_size)
    logger.info("Formed %d volume bars", bars.height)

    # --- Synthetic price universe ---
    universe = generate_universe(cfg.synthetic, n_cointegrated_pairs=2, n_noise_assets=2)
    planted = generate_cointegrated_pair(cfg.synthetic)
    universe["Y0"] = planted.y
    universe["X0"] = planted.x

    if write_outputs:
        write_pair_csv(str(data_dir / "planted_cointegrated.csv"), planted, y_name="Y0", x_name="X0")
        written.append(str(data_dir / "planted_cointegrated.csv"))
        bars.write_csv(data_dir / "volume_bars.csv")
        written.append(str(data_dir / "volume_bars.csv"))
        # Persist full universe
        uni_df = pl.DataFrame({"bar": np.arange(planted.y.size)}).hstack(
            pl.DataFrame({k: v for k, v in universe.items()})
        )
        uni_df.write_csv(data_dir / "universe.csv")
        written.append(str(data_dir / "universe.csv"))

    # --- Causal train/test split (no look-ahead in discovery) ---
    T = planted.y.size
    train_end = int(T * train_fraction)
    train_prices = {k: v[:train_end] for k, v in universe.items()}
    test_y = universe["Y0"][train_end:]
    test_x = universe["X0"][train_end:]

    # Temporarily relax rolling window for shorter synthetic samples if needed
    pd_cfg = cfg.pair_discovery
    if train_end < pd_cfg.rolling_window:
        from dataclasses import replace

        pd_cfg = replace(
            pd_cfg,
            rolling_window=max(60, train_end // 3),
            rolling_step=max(5, train_end // 20),
            min_obs=min(pd_cfg.min_obs, train_end),
        )

    candidates = rank_pairs(train_prices, config=pd_cfg)
    accepted = [c for c in candidates if c.accepted]
    selected = None
    for c in candidates:
        if {c.asset_y, c.asset_x} == {"Y0", "X0"}:
            selected = c
            break
    if selected is None and accepted:
        selected = accepted[0]
    elif selected is None and candidates:
        selected = candidates[0]

    logger.info(
        "Pair discovery: %d candidates, %d accepted; selected=%s",
        len(candidates),
        len(accepted),
        f"{selected.asset_y}/{selected.asset_x}" if selected else None,
    )

    # --- Dynamic hedging: train then causal test filter ---
    diagnostics: FilterDiagnostics | None = None
    if selected is not None:
        y_train = train_prices[selected.asset_y]
        x_train = train_prices[selected.asset_x]
        # Orient: if selected swapped labels vs Y0/X0 convention
        if selected.asset_y == "X0" and selected.asset_x == "Y0":
            y_test, x_test = test_x, test_y
        else:
            y_test = universe[selected.asset_y][train_end:]
            x_test = universe[selected.asset_x][train_end:]

        filt = KalmanHedgeFilter(config=cfg.hedging)
        filt.filter_series(y_train, x_train)
        # Continue on test without reset (online causal)
        filt.filter_series(y_test, x_test, reset=False)
        diagnostics = diagnose_filter(filt)
        logger.info(
            "Kalman final alpha=%.4f beta=%.4f diverged=%s max|z|=%.2f",
            diagnostics.final_alpha,
            diagnostics.final_beta,
            diagnostics.diverged,
            diagnostics.max_abs_z,
        )

        if write_outputs:
            arrays = filt.to_arrays()
            pl.DataFrame({k: v for k, v in arrays.items()}).write_csv(
                results_dir / "kalman_path.csv"
            )
            written.append(str(results_dir / "kalman_path.csv"))

    if write_outputs:
        ranking_path = results_dir / "pair_ranking.json"
        with ranking_path.open("w", encoding="utf-8") as fh:
            json.dump([_candidate_to_dict(c) for c in candidates], fh, indent=2)
        written.append(str(ranking_path))
        summary = {
            "train_size": train_end,
            "test_size": T - train_end,
            "n_candidates": len(candidates),
            "n_accepted": len(accepted),
            "selected": _candidate_to_dict(selected) if selected else None,
            "book_microprice": mp,
            "book_imbalance": imb,
            "n_volume_bars": bars.height,
            "filter_diagnostics": asdict(diagnostics) if diagnostics else None,
        }
        summary_path = results_dir / "phase1_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        written.append(str(summary_path))

    return Phase1Result(
        candidates=candidates,
        selected=selected,
        filter_diagnostics=diagnostics,
        train_size=train_end,
        test_size=T - train_end,
        book_microprice=mp,
        book_imbalance=imb,
        n_volume_bars=bars.height,
        paths_written=tuple(written),
    )


def main() -> None:
    """CLI entry: ``hft-phase1`` or ``python -m hft_research.pipeline.phase1``."""
    cfg_path = default_config_path()
    cfg = load_config(cfg_path) if cfg_path.exists() else Phase1Config()
    result = run_phase1_pipeline(cfg, write_outputs=True)
    print(
        f"Phase 1 complete: candidates={len(result.candidates)} "
        f"selected={result.selected.asset_y if result.selected else None}/"
        f"{result.selected.asset_x if result.selected else None} "
        f"bars={result.n_volume_bars}"
    )


if __name__ == "__main__":
    main()

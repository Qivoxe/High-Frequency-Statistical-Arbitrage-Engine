"""Phase 1 micro-benchmarks (reproducible synthetic workloads)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hft_research.config import HedgingConfig, load_config, default_config_path
from hft_research.hedging.kalman import KalmanHedgeFilter
from hft_research.market_data.bars import aggregate_volume_bars
from hft_research.market_data.order_book import OrderBook
from hft_research.pair_discovery.engle_granger import engle_granger_test
from hft_research.pair_discovery.johansen import johansen_test
from hft_research.pair_discovery.ranking import rank_pairs
from hft_research.synthetic.cointegrated import generate_cointegrated_pair, generate_universe
from hft_research.synthetic.order_book_synth import generate_l2_stream, generate_trades


def _timed(fn, *args, repeats: int = 5, **kwargs) -> dict:
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "repeats": repeats,
        "result_info": out if isinstance(out, (int, float, str, bool)) else None,
    }


def main() -> None:
    cfg = load_config(default_config_path()) if default_config_path().exists() else None
    results: dict = {"benchmarks": {}}

    # Order book reconstruction
    stream = generate_l2_stream(n_events=5000, depth=10, seed=42)

    def recon():
        book = OrderBook(depth=10)
        book.apply_snapshot(stream.snapshot)
        for u in stream.updates:
            book.apply_update(u)
        return book.update_count

    results["benchmarks"]["order_book_5k_updates"] = _timed(recon)

    # Volume bars
    trades = generate_trades(n=20_000, seed=42)

    def bars():
        return aggregate_volume_bars(trades, bar_size=500.0).height

    results["benchmarks"]["volume_bars_20k_trades"] = _timed(bars)

    # Engle-Granger + Johansen
    pair = generate_cointegrated_pair(n=2000, seed=42)

    def eg():
        return engle_granger_test(pair.y, pair.x).residual_adf.pvalue

    def joh():
        return johansen_test(np.column_stack([pair.y, pair.x])).rank_trace

    results["benchmarks"]["engle_granger_2k"] = _timed(eg)
    results["benchmarks"]["johansen_2k"] = _timed(joh)

    # Pair ranking universe
    uni = {k: v[:800] for k, v in generate_universe(seed=42, n_cointegrated_pairs=2, n_noise_assets=3).items()}

    def rank():
        from hft_research.config import PairDiscoveryConfig

        return len(
            rank_pairs(
                uni,
                config=PairDiscoveryConfig(
                    min_obs=100,
                    rolling_window=150,
                    rolling_step=40,
                    min_stability_ratio=0.3,
                ),
            )
        )

    results["benchmarks"]["rank_pairs_5assets_800"] = _timed(rank, repeats=3)

    # Kalman filter
    def kf():
        filt = KalmanHedgeFilter(config=HedgingConfig())
        filt.filter_series(pair.y, pair.x)
        return filt.beta

    results["benchmarks"]["kalman_2k"] = _timed(kf)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Human-readable markdown
    lines = ["# Phase 1 Benchmark Results", ""]
    lines.append("| Benchmark | mean (s) | std (s) | min (s) | max (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in results["benchmarks"].items():
        lines.append(
            f"| `{name}` | {m['mean_s']:.6f} | {m['std_s']:.6f} | {m['min_s']:.6f} | {m['max_s']:.6f} |"
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

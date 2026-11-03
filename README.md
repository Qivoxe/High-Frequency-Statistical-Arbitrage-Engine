# HFT Statistical Arbitrage Research Engine — Phase 1

Institutional-style **high-frequency statistical arbitrage research engine** (Phase 1 core).

Phase 1 delivers market-data infrastructure, pair discovery, and dynamic Kalman hedging with reproducible synthetic datasets and a full automated test suite.

## Features

- **Market data:** L2 order-book reconstruction (snapshot + incremental), sequence validation/recovery, micro-price, order-book imbalance, trade aggregation, deterministic volume bars, timestamp normalization
- **Pair discovery:** Engle–Granger, Johansen (trace & max-eigen), ADF, cointegrating vectors, rolling stability, multiple-testing correction, ranking/filtering
- **Dynamic hedging:** Recursive Kalman filter with time-varying intercept & hedge ratio, innovation covariance, adaptive process noise, diagnostics & divergence detection
- **Engineering:** Modular layout, YAML config, causal train/test split (no look-ahead), synthetic data generators, Pytest coverage

## Requirements

Python **3.12+**, NumPy, Polars, SciPy, Statsmodels, PyYAML, Pytest.

## Install

```bash
cd HFT
python -m pip install -e ".[dev]"
```

## Quick start

```bash
# Run Phase 1 pipeline (writes data/synthetic + results/phase1)
hft-phase1

# Or:
python -m hft_research.pipeline.phase1
```

```bash
# Tests
pytest

# Benchmarks
python benchmarks/run_benchmarks.py
```

## Project layout

```
configs/phase1.yaml          # Central parameters
src/hft_research/
  market_data/               # L2, trades, bars, timestamps
  pair_discovery/            # EG, Johansen, ADF, ranking
  hedging/                   # Kalman + diagnostics
  synthetic/                 # Reproducible generators
  pipeline/phase1.py         # End-to-end orchestration
tests/                       # Unit + integration
docs/                        # Technical docs + Phase 1 report
benchmarks/                  # Timing harness
data/synthetic/              # Generated sample datasets
results/phase1/              # Pipeline outputs
```

## Configuration

All knobs live in [`configs/phase1.yaml`](configs/phase1.yaml). Load via:

```python
from hft_research.config import load_config, default_config_path
cfg = load_config(default_config_path())
```

## Documentation

- [Technical documentation](docs/TECHNICAL.md)
- [Phase 1 completion report](docs/PHASE1_REPORT.md)

## License

MIT

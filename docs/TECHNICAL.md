# Phase 1 Technical Documentation

## Architecture

```
ingestion (market_data)  →  analytics (pair_discovery)  →  hedging (Kalman)
         ↑                            ↑
   synthetic generators          causal train window only
```

Ingestion never imports discovery/hedging. Discovery never mutates books. The Phase 1 pipeline enforces a **train/test split**: pair screening uses only `[:train_end]`; the Kalman filter then runs online through train and continues causally into test without reset.

## Market data

### L2 reconstruction (`OrderBook`)

- Bootstrap via `BookSnapshot`; incremental via `BookUpdate`.
- Sequence: expects `last + 1`. Gaps `> max_sequence_gap` raise `SequenceGapError` and set `needs_snapshot`.
- Stale/duplicate sequences are ignored.
- Crossed books raise `BookIntegrityError`.

### Micro-price

Size-weighted Cont–style micro-price over top-`n` levels:

\[
m = \frac{P^{\text{ask}}_{\text{vw}} V^{\text{bid}} + P^{\text{bid}}_{\text{vw}} V^{\text{ask}}}{V^{\text{bid}} + V^{\text{ask}}}
\]

### Imbalance

\[
I = \frac{V^{\text{bid}} - V^{\text{ask}}}{V^{\text{bid}} + V^{\text{ask}}} \in [-1, 1]
\]

### Volume bars

Bars close when cumulative volume reaches `bar_size`. Excess size from the closing trade **spills** into the next bar at the same price (deterministic, no look-ahead).

### Timestamps

Normalized to UTC nanoseconds. Optional out-of-order drop via `max_out_of_order_ns`.

## Pair discovery

| Test | Role |
|------|------|
| ADF | Marginal stationarity; prefer I(1) legs |
| Engle–Granger | OLS hedge + residual ADF; half-life via OU |
| Johansen | Trace & max-eigen rank; eigenvector (first loading = 1) |
| Rolling EG | Stability ratio & hedge-ratio CV on causal windows |
| BH / Bonferroni | Multiple-testing across pair universe |
| Ranking | Composite score + acceptance filters |

**Look-ahead control:** callers must pass only historical windows. The pipeline does this via `train_fraction`.

**Survivorship:** the synthetic universe is fixed ex ante; production use should freeze the name list at train start.

## Kalman dynamic hedge

State \(\theta_t = [\alpha_t, \beta_t]^\top\) follows a random walk. Observation \(y_t = \alpha_t + \beta_t x_t + v_t\).

- Transition \(F = I\), process noise \(Q\) (optionally adaptive: \(Q \leftarrow Q(1 + \kappa z^2)\)).
- Innovation variance \(S_t = H_t P_{t|t-1} H_t^\top + R\).
- Joseph-form covariance update.
- Divergence: consecutive \(|z| >\) threshold.
- Diagnostics: innovation moments, Ljung–Box whiteness, optional NEES vs synthetic truth.

## Reproducibility

- YAML `synthetic.seed` seeds all RNGs.
- Deterministic bar and book reconstruction given the same event stream.
- Tests pin seeds (`42`, etc.).

## Extension points (Phase 2 — not implemented)

Execution, risk limits, live feeds, portfolio construction. Do not start Phase 2 until Phase 1 tests and benchmarks are green.

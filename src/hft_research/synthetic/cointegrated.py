"""Synthetic cointegrated / non-cointegrated price pair generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hft_research.config import SyntheticConfig


@dataclass(frozen=True, slots=True)
class SyntheticPairResult:
    """Synthetic pair prices with ground-truth parameters."""

    y: np.ndarray
    x: np.ndarray
    spread: np.ndarray
    hedge_ratio: float
    intercept: float
    cointegrated: bool
    seed: int

    def to_frame(self, *, y_name: str = "Y", x_name: str = "X") -> pl.DataFrame:
        return pl.DataFrame(
            {
                "bar": np.arange(self.y.size, dtype=np.int64),
                y_name: self.y,
                x_name: self.x,
                "spread": self.spread,
            }
        )


def generate_cointegrated_pair(
    config: SyntheticConfig | None = None,
    *,
    seed: int | None = None,
    n: int | None = None,
    hedge_ratio: float | None = None,
    intercept: float | None = None,
    kappa: float | None = None,
) -> SyntheticPairResult:
    """Generate an I(1) pair with a mean-reverting cointegrating residual.

    Construction
    ------------
    1. ``x_t`` is a random walk.
    2. Spread ``s_t`` follows OU / AR(1): ``s_t = (1-κ) s_{t-1} + ε``.
    3. ``y_t = α + β x_t + s_t``.

    This guarantees cointegration with known ``(α, β)`` under κ > 0.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    n_obs = int(n if n is not None else cfg.n_bars)
    beta = float(hedge_ratio if hedge_ratio is not None else cfg.hedge_ratio_true)
    alpha = float(intercept if intercept is not None else cfg.intercept_true)
    k = float(kappa if kappa is not None else cfg.cointegration_strength)
    if n_obs < 2:
        raise ValueError("n must be >= 2")
    if not (0.0 < k < 2.0):
        raise ValueError("kappa must be in (0, 2)")

    shocks_x = rng.normal(0.0, cfg.price_vol, size=n_obs)
    x = 100.0 + np.cumsum(shocks_x)

    spread = np.zeros(n_obs, dtype=np.float64)
    eps = rng.normal(0.0, cfg.spread_vol, size=n_obs)
    for t in range(1, n_obs):
        spread[t] = (1.0 - k) * spread[t - 1] + eps[t]

    y = alpha + beta * x + spread
    return SyntheticPairResult(
        y=y,
        x=x,
        spread=spread,
        hedge_ratio=beta,
        intercept=alpha,
        cointegrated=True,
        seed=int(seed if seed is not None else cfg.seed),
    )


def generate_independent_pair(
    config: SyntheticConfig | None = None,
    *,
    seed: int | None = None,
    n: int | None = None,
) -> SyntheticPairResult:
    """Generate two independent random walks (not cointegrated)."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed + 1)
    n_obs = int(n if n is not None else cfg.n_bars)
    x = 100.0 + np.cumsum(rng.normal(0.0, cfg.price_vol, size=n_obs))
    y = 100.0 + np.cumsum(rng.normal(0.0, cfg.price_vol, size=n_obs))
    spread = y - x
    return SyntheticPairResult(
        y=y,
        x=x,
        spread=spread,
        hedge_ratio=1.0,
        intercept=0.0,
        cointegrated=False,
        seed=int(seed if seed is not None else cfg.seed + 1),
    )


def generate_universe(
    config: SyntheticConfig | None = None,
    *,
    n_cointegrated_pairs: int = 2,
    n_noise_assets: int = 3,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Build a small universe with planted cointegrated pairs and noise RW.

    Returns a mapping of asset name -> price path. Planted pairs are
    ``(Y0, X0), (Y1, X1), ...``. Noise assets ``N0, N1, ...`` are independent.
    """
    cfg = config or SyntheticConfig()
    base_seed = seed if seed is not None else cfg.seed
    out: dict[str, np.ndarray] = {}
    for i in range(n_cointegrated_pairs):
        pair = generate_cointegrated_pair(
            cfg,
            seed=base_seed + 100 * i,
            hedge_ratio=cfg.hedge_ratio_true + 0.1 * i,
        )
        out[f"Y{i}"] = pair.y
        out[f"X{i}"] = pair.x
    for j in range(n_noise_assets):
        noise = generate_independent_pair(cfg, seed=base_seed + 1000 + j)
        out[f"N{j}"] = noise.y
    return out


def write_pair_csv(path: str, result: SyntheticPairResult, *, y_name: str = "Y", x_name: str = "X") -> None:
    """Persist a synthetic pair to CSV for offline inspection."""
    result.to_frame(y_name=y_name, x_name=x_name).write_csv(path)

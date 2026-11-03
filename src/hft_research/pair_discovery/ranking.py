"""Pair ranking and filtering for statistical arbitrage universes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from hft_research.config import PairDiscoveryConfig
from hft_research.exceptions import InsufficientDataError, PairDiscoveryError
from hft_research.logging_utils import get_logger
from hft_research.pair_discovery.engle_granger import engle_granger_test
from hft_research.pair_discovery.johansen import johansen_test
from hft_research.pair_discovery.multiple_testing import correct_pvalues
from hft_research.pair_discovery.stability import rolling_cointegration_stability
from hft_research.pair_discovery.stationarity import adf_test

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PairCandidate:
    """Ranked pair candidate with diagnostics (no look-ahead beyond sample)."""

    asset_y: str
    asset_x: str
    eg_pvalue: float
    eg_pvalue_corrected: float
    johansen_cointegrated: bool
    johansen_rank: int
    hedge_ratio: float
    intercept: float
    half_life: float
    residual_std: float
    stability_ratio: float
    hedge_ratio_cv: float
    y_adf_pvalue: float
    x_adf_pvalue: float
    score: float
    accepted: bool


def _pair_score(
    *,
    eg_pvalue_corrected: float,
    johansen_cointegrated: bool,
    half_life: float,
    stability_ratio: float,
    hedge_ratio_cv: float,
    half_life_min: float,
    half_life_max: float,
) -> float:
    """Higher score = more attractive mean-reverting pair.

    Components (all in [0, 1] after transforms):
    - cointegration strength (1 - corrected p)
    - Johansen confirmation bonus
    - half-life preference (mid-range preferred)
    - rolling stability
    - hedge-ratio stability (low CV)
    """
    coint_strength = max(0.0, 1.0 - float(eg_pvalue_corrected))
    johansen_bonus = 0.15 if johansen_cointegrated else 0.0

    if not np.isfinite(half_life) or half_life <= 0:
        hl_score = 0.0
    elif half_life < half_life_min or half_life > half_life_max:
        hl_score = 0.1
    else:
        # Peak at geometric mid of admissible band
        mid = np.sqrt(half_life_min * half_life_max)
        hl_score = float(np.exp(-0.5 * ((np.log(half_life) - np.log(mid)) / 0.75) ** 2))

    stab = float(np.clip(stability_ratio, 0.0, 1.0))
    cv_score = float(np.exp(-max(hedge_ratio_cv, 0.0)))

    return (
        0.35 * coint_strength
        + johansen_bonus
        + 0.20 * hl_score
        + 0.20 * stab
        + 0.10 * cv_score
    )


def rank_pairs(
    prices: Mapping[str, np.ndarray] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
    config: PairDiscoveryConfig | None = None,
) -> list[PairCandidate]:
    """Screen all unordered pairs and return ranked candidates.

    Parameters
    ----------
    prices:
        Mapping of asset name -> price series, or 2-D array ``(T, N)``.
    names:
        Column names when ``prices`` is an array.
    config:
        Screening parameters. Defaults applied if omitted.

    Notes
    -----
    All tests use only the provided sample. Callers must supply a training
    window that ends before the evaluation period to avoid look-ahead and
    survivorship bias (do not drop delisted names after peeking at future
    returns).
    """
    cfg = config or PairDiscoveryConfig()

    if isinstance(prices, Mapping):
        names_list = list(prices.keys())
        series = [np.asarray(prices[n], dtype=np.float64).ravel() for n in names_list]
    else:
        arr = np.asarray(prices, dtype=np.float64)
        if arr.ndim != 2:
            raise PairDiscoveryError("prices array must be 2-D (T, N)")
        if names is None:
            names_list = [f"S{i}" for i in range(arr.shape[1])]
        else:
            names_list = list(names)
            if len(names_list) != arr.shape[1]:
                raise PairDiscoveryError("names length must match columns")
        series = [arr[:, i] for i in range(arr.shape[1])]

    n_assets = len(names_list)
    if n_assets < 2:
        raise PairDiscoveryError("Need at least 2 assets")

    lengths = {s.size for s in series}
    if len(lengths) != 1:
        raise PairDiscoveryError("All series must share the same length")
    T = series[0].size
    if T < cfg.min_obs:
        raise InsufficientDataError(f"Need >= {cfg.min_obs} obs, got {T}")

    # Precompute marginal ADF (prefer I(1) legs: fail to reject unit root)
    adf_p: dict[str, float] = {}
    for name, s in zip(names_list, series, strict=True):
        try:
            adf_p[name] = adf_test(s, significance=cfg.adf_significance).pvalue
        except InsufficientDataError:
            adf_p[name] = 0.0  # treat as stationary / unusable

    raw_rows: list[dict] = []
    for i, j in combinations(range(n_assets), 2):
        y_name, x_name = names_list[i], names_list[j]
        y, x = series[i], series[j]
        try:
            eg = engle_granger_test(y, x, significance=cfg.eg_significance)
            joh = johansen_test(
                np.column_stack([y, x]),
                det_order=cfg.johansen_det_order,
                k_ar_diff=cfg.johansen_k_ar_diff,
                significance=cfg.johansen_significance,
            )
            if T >= cfg.rolling_window:
                stab = rolling_cointegration_stability(
                    y,
                    x,
                    window=cfg.rolling_window,
                    step=cfg.rolling_step,
                    significance=cfg.eg_significance,
                )
                stab_ratio = stab.stability_ratio
                hr_cv = stab.hedge_ratio_cv
            else:
                stab_ratio = 1.0 if eg.is_cointegrated else 0.0
                hr_cv = 0.0
        except (InsufficientDataError, PairDiscoveryError) as exc:
            logger.debug("Skipping pair %s/%s: %s", y_name, x_name, exc)
            continue

        raw_rows.append(
            {
                "asset_y": y_name,
                "asset_x": x_name,
                "eg_pvalue": eg.residual_adf.pvalue,
                "johansen_cointegrated": joh.is_cointegrated,
                "johansen_rank": max(joh.rank_trace, joh.rank_max_eig),
                "hedge_ratio": eg.hedge_ratio,
                "intercept": eg.intercept,
                "half_life": eg.half_life,
                "residual_std": eg.residual_std,
                "stability_ratio": stab_ratio,
                "hedge_ratio_cv": hr_cv,
                "y_adf_pvalue": adf_p[y_name],
                "x_adf_pvalue": adf_p[x_name],
            }
        )

    if not raw_rows:
        return []

    pvals = np.array([r["eg_pvalue"] for r in raw_rows], dtype=np.float64)
    mt = correct_pvalues(
        pvals,
        method=cfg.multiple_testing,
        alpha=cfg.fdr_alpha,
    )

    candidates: list[PairCandidate] = []
    for idx, row in enumerate(raw_rows):
        p_corr = float(mt.corrected_pvalues[idx])
        score = _pair_score(
            eg_pvalue_corrected=p_corr,
            johansen_cointegrated=bool(row["johansen_cointegrated"]),
            half_life=float(row["half_life"]),
            stability_ratio=float(row["stability_ratio"]),
            hedge_ratio_cv=float(row["hedge_ratio_cv"]),
            half_life_min=cfg.half_life_min,
            half_life_max=cfg.half_life_max,
        )
        # Prefer I(1) legs: high ADF p-values on levels
        legs_ok = row["y_adf_pvalue"] >= cfg.adf_significance and row["x_adf_pvalue"] >= cfg.adf_significance
        hl_ok = (
            np.isfinite(row["half_life"])
            and cfg.half_life_min <= row["half_life"] <= cfg.half_life_max
        )
        accepted = bool(
            mt.rejected[idx]
            and row["johansen_cointegrated"]
            and row["stability_ratio"] >= cfg.min_stability_ratio
            and hl_ok
            and legs_ok
        )
        candidates.append(
            PairCandidate(
                asset_y=row["asset_y"],
                asset_x=row["asset_x"],
                eg_pvalue=float(row["eg_pvalue"]),
                eg_pvalue_corrected=p_corr,
                johansen_cointegrated=bool(row["johansen_cointegrated"]),
                johansen_rank=int(row["johansen_rank"]),
                hedge_ratio=float(row["hedge_ratio"]),
                intercept=float(row["intercept"]),
                half_life=float(row["half_life"]),
                residual_std=float(row["residual_std"]),
                stability_ratio=float(row["stability_ratio"]),
                hedge_ratio_cv=float(row["hedge_ratio_cv"]),
                y_adf_pvalue=float(row["y_adf_pvalue"]),
                x_adf_pvalue=float(row["x_adf_pvalue"]),
                score=score,
                accepted=accepted,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    logger.info(
        "Ranked %d pairs; %d accepted after filters",
        len(candidates),
        sum(1 for c in candidates if c.accepted),
    )
    return candidates

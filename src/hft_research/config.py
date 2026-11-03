"""Configuration loading and typed access for Phase 1 and Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from hft_research.exceptions import ConfigurationError


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"Missing required config key: {key}")
    return mapping[key]


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    book_depth: int = 10
    max_sequence_gap: int = 1
    volume_bar_size: float = 1000.0
    timestamp_unit: str = "ns"
    max_out_of_order_ns: int = 0
    top_n_levels: int = 5


@dataclass(frozen=True, slots=True)
class PairDiscoveryConfig:
    min_obs: int = 100
    adf_significance: float = 0.05
    eg_significance: float = 0.05
    johansen_significance: float = 0.05
    johansen_det_order: str = "const"
    johansen_k_ar_diff: int = 1
    johansen_max_lags: int = 5
    rolling_window: int = 252
    rolling_step: int = 21
    min_stability_ratio: float = 0.60
    multiple_testing: str = "benjamini_hochberg"
    fdr_alpha: float = 0.05
    half_life_min: float = 1.0
    half_life_max: float = 100.0


@dataclass(frozen=True, slots=True)
class HedgingConfig:
    initial_P_diag: tuple[float, float] = (1.0e4, 1.0e2)
    observation_noise: float = 1.0e-4
    process_noise_diag: tuple[float, float] = (1.0e-6, 1.0e-7)
    adaptive_process_noise: bool = True
    adaptive_kappa: float = 0.05
    divergence_z: float = 5.0
    divergence_streak: int = 10
    nees_upper: float = 9.21


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    seed: int = 42
    n_bars: int = 2000
    n_book_events: int = 5000
    cointegration_strength: float = 0.15
    hedge_ratio_true: float = 1.25
    intercept_true: float = 0.5
    price_vol: float = 0.01
    spread_vol: float = 0.002
    tick_size: float = 0.01
    lot_size: float = 1.0


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


@dataclass(frozen=True, slots=True)
class PathsConfig:
    data_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    results_dir: Path = field(default_factory=lambda: Path("results/phase1"))
    benchmark_dir: Path = field(default_factory=lambda: Path("benchmarks"))


@dataclass(frozen=True, slots=True)
class OUConfig:
    min_obs: int = 50
    theta_min: float = 1.0e-6
    theta_max: float = 50.0
    sigma_min: float = 1.0e-8
    sigma_max: float = 10.0
    max_iter: int = 500
    gtol: float = 1.0e-8
    require_stationary: bool = True
    half_life_min: float = 0.5
    half_life_max: float = 500.0
    residual_ljung_box_alpha: float = 0.01
    conf_level: float = 0.95


@dataclass(frozen=True, slots=True)
class SpreadConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    neutral_z: float = 0.25
    rolling_window: int = 60
    use_model_z: bool = True
    max_holding_bars: int = 100
    min_bars_between_flips: int = 3
    warmup_bars: int = 30


@dataclass(frozen=True, slots=True)
class CostsConfig:
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 3.0
    default_is_maker: bool = False
    bid_ask_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    impact_coeff_bps: float = 5.0
    borrow_rate_annual: float = 0.02
    funding_rate_annual: float = 0.0
    latency_bps: float = 0.5
    bars_per_year: float = 252.0
    min_net_edge: float = 0.0
    n_legs: int = 2


@dataclass(frozen=True, slots=True)
class RiskConfig:
    target_vol: float = 0.10
    max_gross_exposure: float = 1.0e6
    max_net_exposure: float = 5.0e5
    max_pair_inventory: float = 1.0e5
    max_portfolio_pairs: int = 10
    max_drawdown: float = 0.10
    stop_loss: float = 0.05
    spread_collapse_z: float = 0.05
    spread_collapse_var_ratio: float = 0.05
    theta_instability_ratio: float = 0.25
    enable_kill_switch: bool = True


@dataclass(frozen=True, slots=True)
class Phase1Config:
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    pair_discovery: PairDiscoveryConfig = field(default_factory=PairDiscoveryConfig)
    hedging: HedgingConfig = field(default_factory=HedgingConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Phase1Config:
        """Build a typed Phase 1 config from a nested mapping (e.g. YAML)."""
        md = dict(_require(raw, "market_data"))
        pd = dict(_require(raw, "pair_discovery"))
        hg = dict(_require(raw, "hedging"))
        syn = dict(_require(raw, "synthetic"))
        log = dict(raw.get("logging", {}))
        paths = dict(raw.get("paths", {}))

        hedge = HedgingConfig(
            initial_P_diag=tuple(hg.get("initial_P_diag", (1.0e4, 1.0e2))),  # type: ignore[arg-type]
            observation_noise=float(hg.get("observation_noise", 1.0e-4)),
            process_noise_diag=tuple(hg.get("process_noise_diag", (1.0e-6, 1.0e-7))),  # type: ignore[arg-type]
            adaptive_process_noise=bool(hg.get("adaptive_process_noise", True)),
            adaptive_kappa=float(hg.get("adaptive_kappa", 0.05)),
            divergence_z=float(hg.get("divergence_z", 5.0)),
            divergence_streak=int(hg.get("divergence_streak", 10)),
            nees_upper=float(hg.get("nees_upper", 9.21)),
        )
        path_cfg = PathsConfig(
            data_dir=Path(paths.get("data_dir", "data/synthetic")),
            results_dir=Path(paths.get("results_dir", "results/phase1")),
            benchmark_dir=Path(paths.get("benchmark_dir", "benchmarks")),
        )
        return cls(
            market_data=MarketDataConfig(**md),
            pair_discovery=PairDiscoveryConfig(**pd),
            hedging=hedge,
            synthetic=SyntheticConfig(**syn),
            logging=LoggingConfig(**log) if log else LoggingConfig(),
            paths=path_cfg,
        )


@dataclass(frozen=True, slots=True)
class Phase2Config:
    """Phase 2 research config embedding Phase 1 settings."""

    phase1: Phase1Config = field(default_factory=Phase1Config)
    ou: OUConfig = field(default_factory=OUConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paths: PathsConfig = field(
        default_factory=lambda: PathsConfig(results_dir=Path("results/phase2"))
    )

    @property
    def market_data(self) -> MarketDataConfig:
        return self.phase1.market_data

    @property
    def pair_discovery(self) -> PairDiscoveryConfig:
        return self.phase1.pair_discovery

    @property
    def hedging(self) -> HedgingConfig:
        return self.phase1.hedging

    @property
    def synthetic(self) -> SyntheticConfig:
        return self.phase1.synthetic

    @property
    def logging(self) -> LoggingConfig:
        return self.phase1.logging

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> Phase2Config:
        """Build Phase 2 config, optionally merging an included Phase 1 YAML."""
        merged = dict(raw)
        include = merged.pop("include_phase1", None)
        if include is not None:
            inc_path = Path(include)
            if not inc_path.is_absolute():
                root = base_dir or Path.cwd()
                # Try relative to config file dir, then repo configs/
                candidates = [
                    root / inc_path,
                    Path(__file__).resolve().parents[2] / inc_path,
                ]
                for c in candidates:
                    if c.exists():
                        inc_path = c
                        break
            if not inc_path.exists():
                raise ConfigurationError(f"include_phase1 not found: {include}")
            with inc_path.open("r", encoding="utf-8") as fh:
                phase1_raw = yaml.safe_load(fh)
            if not isinstance(phase1_raw, dict):
                raise ConfigurationError("Included Phase 1 config must be a mapping")
            merged = _deep_merge(phase1_raw, merged)

        # Phase 1 subsection (may reuse top-level keys)
        p1 = Phase1Config.from_mapping(merged)
        ou_raw = dict(merged.get("ou", {}))
        sp_raw = dict(merged.get("spread", {}))
        cost_raw = dict(merged.get("costs", {}))
        risk_raw = dict(merged.get("risk", {}))
        paths_raw = dict(merged.get("paths", {}))

        # Validate hysteresis
        entry_z = float(sp_raw.get("entry_z", 2.0))
        exit_z = float(sp_raw.get("exit_z", 0.5))
        if entry_z <= exit_z:
            raise ConfigurationError("spread.entry_z must be > spread.exit_z (hysteresis)")

        path_cfg = PathsConfig(
            data_dir=Path(paths_raw.get("data_dir", p1.paths.data_dir)),
            results_dir=Path(paths_raw.get("results_dir", "results/phase2")),
            benchmark_dir=Path(paths_raw.get("benchmark_dir", p1.paths.benchmark_dir)),
        )
        # Keep Phase 1 results_dir for Phase 1 artifacts unless overridden only in phase2 paths
        p1 = replace(p1, paths=replace(p1.paths, data_dir=path_cfg.data_dir))

        return cls(
            phase1=p1,
            ou=OUConfig(**ou_raw) if ou_raw else OUConfig(),
            spread=SpreadConfig(**{**{f.name: getattr(SpreadConfig(), f.name) for f in fields(SpreadConfig)}, **sp_raw}),
            costs=CostsConfig(**cost_raw) if cost_raw else CostsConfig(),
            risk=RiskConfig(**risk_raw) if risk_raw else RiskConfig(),
            paths=path_cfg,
        )


def load_config(path: str | Path | None = None) -> Phase1Config:
    """Load Phase 1 configuration from YAML, or return defaults."""
    if path is None:
        return Phase1Config()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigurationError("Config root must be a mapping")
    return Phase1Config.from_mapping(raw)


def load_phase2_config(path: str | Path | None = None) -> Phase2Config:
    """Load Phase 2 configuration (with optional Phase 1 include)."""
    if path is None:
        return Phase2Config()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigurationError("Config root must be a mapping")
    return Phase2Config.from_mapping(raw, base_dir=cfg_path.parent)


def default_config_path() -> Path:
    """Resolve the repository default Phase 1 config path."""
    return Path(__file__).resolve().parents[2] / "configs" / "phase1.yaml"


def default_phase2_config_path() -> Path:
    """Resolve the repository default Phase 2 config path."""
    return Path(__file__).resolve().parents[2] / "configs" / "phase2.yaml"

"""Domain-specific exceptions for the Phase 1 research engine."""

from __future__ import annotations


class HFTResearchError(Exception):
    """Base exception for all research-engine failures."""


class ConfigurationError(HFTResearchError):
    """Invalid or incomplete configuration."""


class MarketDataError(HFTResearchError):
    """Market-data ingestion or reconstruction failure."""


class SequenceGapError(MarketDataError):
    """Detected gap in order-book update sequence numbers."""

    def __init__(self, expected: int, received: int, message: str | None = None) -> None:
        self.expected = expected
        self.received = received
        detail = message or f"Sequence gap: expected {expected}, received {received}"
        super().__init__(detail)


class BookIntegrityError(MarketDataError):
    """Order book crossed, empty, or otherwise inconsistent."""


class TimestampError(MarketDataError):
    """Invalid or non-normalizable timestamps."""


class PairDiscoveryError(HFTResearchError):
    """Cointegration / stationarity testing failure."""


class InsufficientDataError(HFTResearchError):
    """Not enough observations for the requested statistical procedure."""


class KalmanFilterError(HFTResearchError):
    """Kalman filter numerical or divergence failure."""


class LookAheadBiasError(HFTResearchError):
    """Attempt to use future information in a causal pipeline step."""


class OUCalibrationError(HFTResearchError):
    """Ornstein–Uhlenbeck MLE or diagnostics failure."""


class SpreadEngineError(HFTResearchError):
    """Spread construction or signal-generation failure."""


class TransactionCostError(HFTResearchError):
    """Transaction-cost evaluation failure."""


class RiskLimitError(HFTResearchError):
    """Hard risk-limit violation or kill-switch trip."""


class RiskEngineError(HFTResearchError):
    """Risk engine configuration or state failure."""

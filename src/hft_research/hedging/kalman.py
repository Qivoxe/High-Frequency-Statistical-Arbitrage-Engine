"""Recursive Kalman filter for time-varying intercept and hedge ratio.

State-space model
-----------------
State (random walk)::

    θ_t = [α_t, β_t]^T
    θ_t = θ_{t-1} + w_t,   w_t ~ N(0, Q_t)

Observation::

    y_t = α_t + β_t x_t + v_t = H_t θ_t + v_t,
    H_t = [1, x_t],   v_t ~ N(0, R)

Innovation covariance::

    S_t = H_t P_{t|t-1} H_t^T + R

Adaptive process noise (optional)::

    Q_t = Q_base * (1 + κ z_t^2),  z_t = ν_t / sqrt(S_t)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hft_research.config import HedgingConfig
from hft_research.exceptions import KalmanFilterError
from hft_research.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class KalmanState:
    """Filtered state at one time step (posterior)."""

    t: int
    alpha: float
    beta: float
    P: np.ndarray
    innovation: float
    innovation_var: float
    kalman_gain: np.ndarray
    z_score: float
    Q_used: np.ndarray


@dataclass
class KalmanHedgeFilter:
    """Online Kalman filter estimating ``y = α + β x + ε`` with TVP α, β."""

    config: HedgingConfig = field(default_factory=HedgingConfig)
    _theta: np.ndarray = field(init=False, repr=False)
    _P: np.ndarray = field(init=False, repr=False)
    _Q_base: np.ndarray = field(init=False, repr=False)
    _R: float = field(init=False, repr=False)
    _t: int = field(init=False, default=0, repr=False)
    _history: list[KalmanState] = field(init=False, default_factory=list, repr=False)
    _div_streak: int = field(init=False, default=0, repr=False)
    _diverged: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, theta0: np.ndarray | None = None) -> None:
        """Reset filter to prior. Optional ``theta0 = [alpha, beta]``."""
        cfg = self.config
        if theta0 is None:
            self._theta = np.array([0.0, 1.0], dtype=np.float64)
        else:
            self._theta = np.asarray(theta0, dtype=np.float64).ravel()
            if self._theta.shape != (2,):
                raise KalmanFilterError("theta0 must have shape (2,)")
        self._P = np.diag(np.asarray(cfg.initial_P_diag, dtype=np.float64))
        self._Q_base = np.diag(np.asarray(cfg.process_noise_diag, dtype=np.float64))
        self._R = float(cfg.observation_noise)
        if self._R <= 0:
            raise KalmanFilterError("observation_noise must be positive")
        self._t = 0
        self._history.clear()
        self._div_streak = 0
        self._diverged = False

    @property
    def diverged(self) -> bool:
        return self._diverged

    @property
    def history(self) -> list[KalmanState]:
        return list(self._history)

    @property
    def alpha(self) -> float:
        return float(self._theta[0])

    @property
    def beta(self) -> float:
        return float(self._theta[1])

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    def predict(self, Q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Time update: random-walk state transition ``F = I``."""
        Q_use = self._Q_base if Q is None else Q
        theta_pred = self._theta.copy()  # F = I
        P_pred = self._P + Q_use
        return theta_pred, P_pred

    def update(self, y: float, x: float) -> KalmanState:
        """One predict–update step for observation ``(y, x)``.

        Returns the posterior :class:`KalmanState`. Raises
        :class:`KalmanFilterError` if numerical failure occurs.
        """
        if not np.isfinite(y) or not np.isfinite(x):
            raise KalmanFilterError("Non-finite observation")

        cfg = self.config
        H = np.array([1.0, float(x)], dtype=np.float64)

        theta_pred, P_pred = self.predict(self._Q_base)
        S = float(H @ P_pred @ H.T + self._R)
        if S <= 0 or not np.isfinite(S):
            raise KalmanFilterError(f"Invalid innovation variance S={S}")
        nu = float(y - H @ theta_pred)
        z = nu / np.sqrt(S)

        if cfg.adaptive_process_noise:
            scale = 1.0 + cfg.adaptive_kappa * (z ** 2)
            Q_used = self._Q_base * scale
            theta_pred, P_pred = self.predict(Q_used)
            S = float(H @ P_pred @ H.T + self._R)
            if S <= 0 or not np.isfinite(S):
                raise KalmanFilterError(f"Invalid adaptive innovation variance S={S}")
            nu = float(y - H @ theta_pred)
            z = nu / np.sqrt(S)
        else:
            Q_used = self._Q_base.copy()

        K = (P_pred @ H) / S
        self._theta = theta_pred + K * nu
        # Joseph form for numerical stability
        eye = np.eye(2)
        KH = np.outer(K, H)
        self._P = (eye - KH) @ P_pred @ (eye - KH).T + np.outer(K, K) * self._R
        self._P = 0.5 * (self._P + self._P.T)

        if abs(z) > cfg.divergence_z:
            self._div_streak += 1
        else:
            self._div_streak = 0
        if self._div_streak >= cfg.divergence_streak:
            self._diverged = True
            logger.warning(
                "Kalman filter divergence: |z|>%.2f for %d consecutive steps",
                cfg.divergence_z,
                self._div_streak,
            )

        state = KalmanState(
            t=self._t,
            alpha=float(self._theta[0]),
            beta=float(self._theta[1]),
            P=self._P.copy(),
            innovation=nu,
            innovation_var=S,
            kalman_gain=K.copy(),
            z_score=float(z),
            Q_used=Q_used.copy(),
        )
        self._history.append(state)
        self._t += 1
        return state

    def filter_series(
        self,
        y: np.ndarray,
        x: np.ndarray,
        *,
        reset: bool = True,
        theta0: np.ndarray | None = None,
    ) -> list[KalmanState]:
        """Run the filter over aligned series ``y``, ``x`` (causal online)."""
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        if y_arr.size != x_arr.size:
            raise KalmanFilterError("y and x length mismatch")
        if reset:
            self.reset(theta0=theta0)
        states: list[KalmanState] = []
        for yt, xt in zip(y_arr, x_arr, strict=True):
            if not np.isfinite(yt) or not np.isfinite(xt):
                continue
            states.append(self.update(float(yt), float(xt)))
        return states

    def spread(self, y: float, x: float) -> float:
        """Current residual spread using latest filtered hedge."""
        return float(y - self.alpha - self.beta * x)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Export filter history as NumPy arrays."""
        h = self._history
        if not h:
            return {
                "t": np.array([], dtype=np.int64),
                "alpha": np.array([], dtype=np.float64),
                "beta": np.array([], dtype=np.float64),
                "innovation": np.array([], dtype=np.float64),
                "innovation_var": np.array([], dtype=np.float64),
                "z_score": np.array([], dtype=np.float64),
            }
        return {
            "t": np.array([s.t for s in h], dtype=np.int64),
            "alpha": np.array([s.alpha for s in h], dtype=np.float64),
            "beta": np.array([s.beta for s in h], dtype=np.float64),
            "innovation": np.array([s.innovation for s in h], dtype=np.float64),
            "innovation_var": np.array([s.innovation_var for s in h], dtype=np.float64),
            "z_score": np.array([s.z_score for s in h], dtype=np.float64),
            "P00": np.array([s.P[0, 0] for s in h], dtype=np.float64),
            "P11": np.array([s.P[1, 1] for s in h], dtype=np.float64),
        }

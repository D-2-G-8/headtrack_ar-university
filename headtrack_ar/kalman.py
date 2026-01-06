"""
Simple Kalman filter for 2D position + log-scale with constant velocity.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


class KalmanFace2DScale:
    """Kalman filter for face center (cx, cy) and log-scale (s)."""

    def __init__(
        self,
        x0: Iterable[float],
        P0: np.ndarray,
        *,
        q_pos: float = 15.0,
        q_vel: float = 50.0,
        q_scale: float = 0.08,
        q_scale_vel: float = 0.25,
        r_pos_base: float = 25.0,
        r_scale_base: float = 0.12,
        r_conf_floor: float = 0.2,
        gate_threshold: float = 9.0
    ) -> None:
        x0 = np.asarray(list(x0), dtype=float).reshape(6, 1)
        if x0.shape != (6, 1):
            raise ValueError("x0 must be length-6")
        self.x = x0
        self.P = np.asarray(P0, dtype=float).reshape(6, 6)
        if self.P.shape != (6, 6):
            raise ValueError("P0 must be 6x6")
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.q_scale = float(q_scale)
        self.q_scale_vel = float(q_scale_vel)
        self.r_pos_base = float(r_pos_base)
        self.r_scale_base = float(r_scale_base)
        self.r_conf_floor = float(r_conf_floor)
        self.gate_threshold = float(gate_threshold)
        self.last_d2: Optional[float] = None

    def predict(self, dt: float) -> None:
        """Predict state forward by dt seconds."""
        dt = float(max(dt, 1e-3))
        F = np.array(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float
        )
        q = np.diag(
            [
                self.q_pos * dt,
                self.q_pos * dt,
                self.q_scale * dt,
                self.q_vel * dt,
                self.q_vel * dt,
                self.q_scale_vel * dt,
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q

    def update(
        self,
        z: Iterable[float],
        conf: Optional[float],
        *,
        gate_mode: Optional[str] = None,
        gate_threshold: Optional[float] = None,
        r_pos_base: Optional[float] = None,
        r_scale_base: Optional[float] = None,
        r_conf_floor: Optional[float] = None,
        force_accept: bool = False
    ) -> Tuple[bool, float]:
        """Update with measurement z=[cx, cy, s]. Returns (accepted, d2)."""
        z = np.asarray(list(z), dtype=float).reshape(3, 1)
        if z.shape != (3, 1):
            raise ValueError("z must be length-3")
        conf_value = float(conf) if conf is not None else 0.0
        conf_floor = self.r_conf_floor if r_conf_floor is None else float(r_conf_floor)
        conf_value = max(conf_value, conf_floor)
        pos_base = self.r_pos_base if r_pos_base is None else float(r_pos_base)
        scale_base = self.r_scale_base if r_scale_base is None else float(r_scale_base)
        gate = self.gate_threshold if gate_threshold is None else float(gate_threshold)
        mode = (gate_mode or "xys").lower()
        if mode not in ("xy", "xys"):
            mode = "xys"
        r_pos = pos_base / conf_value
        r_scale = scale_base / conf_value
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float
        )
        R = np.diag([r_pos, r_pos, r_scale])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        if mode == "xy":
            y_xy = y[:2, :]
            S_xy = S[:2, :2]
            try:
                S_xy_inv = np.linalg.inv(S_xy)
            except np.linalg.LinAlgError:
                S_xy_inv = np.linalg.pinv(S_xy)
            d2 = float((y_xy.T @ S_xy_inv @ y_xy)[0, 0])
        else:
            d2 = float((y.T @ S_inv @ y)[0, 0])
        self.last_d2 = d2
        if not force_accept and d2 > gate:
            return False, d2
        K = self.P @ H.T @ S_inv
        I = np.eye(self.P.shape[0])
        self.x = self.x + K @ y
        self.P = (I - K @ H) @ self.P
        return True, d2

    def get_state(self) -> tuple[float, float, float, float, float, float]:
        """Return state vector (cx, cy, s, vcx, vcy, vs)."""
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0]),
            float(self.x[4, 0]),
            float(self.x[5, 0]),
        )

    def get_uncertainty(self) -> tuple[float, float, float]:
        """Return std dev for (cx, cy, s)."""
        def _sigma(idx: int) -> float:
            value = float(self.P[idx, idx])
            return float(np.sqrt(value)) if value > 0.0 else 0.0
        return (_sigma(0), _sigma(1), _sigma(2))

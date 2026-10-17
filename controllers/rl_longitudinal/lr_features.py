"""Fixed-length policy inputs from ``Observation.road`` and the planner output.

The order is part of the saved policy. Do not reorder without retraining.
"""
from __future__ import annotations

import math

import numpy as np

from lr_planner import G, Plan, PlannerParams, pedal_caps

PREVIEW_D = (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0, 40.0)
CURV_D = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0)
V_SCALE = 30.0

FEATURE_NAMES = (
    ("speed", "ax", "ay", "throttle", "brake")
    + tuple(f"vsoft_{int(d)}" for d in PREVIEW_D)
    + ("gap_soft", "gap_hard", "slack_soft", "slack_hard")
    + tuple(f"kappa_{int(d)}" for d in CURV_D)
    + ("lateral_offset", "heading_error")
    + ("obstacle_dist", "obstacle_present", "obstacle_shift")
    + ("finish_dist", "finish_present")
    + ("cap_throttle", "cap_brake")
)
FEATURE_DIM = len(FEATURE_NAMES)


def _clip(x, lo, hi) -> float:
    return float(np.clip(x, lo, hi))


def _slack(s, v_ceiling, v, a_dec) -> float:
    """Distance spare before braking at ``a_dec`` is needed to meet the ceiling everywhere ahead."""
    need = np.maximum(v * v - v_ceiling * v_ceiling, 0.0) / (2.0 * max(a_dec, 0.5))
    return float(np.min(s - need))


def featurize(obs, vehicle, pl: Plan, prm: PlannerParams | None = None) -> np.ndarray:
    prm = prm or PlannerParams()
    mu_g = max(vehicle.mu * G, 1e-3)
    v = max(float(obs.speed), 0.0)
    cap_t, cap_b = pedal_caps(obs, vehicle, prm)
    base = [
        _clip(v / V_SCALE, 0.0, 1.5), _clip(obs.ax / 10.0, -1.5, 1.5), _clip(obs.ay / mu_g, -1.5, 1.5),
        _clip(obs.throttle, 0.0, 1.0), _clip(obs.brake, 0.0, 1.0),
    ]
    if not pl.valid:
        return np.asarray(base + [0.0] * (FEATURE_DIM - len(base) - 2) + [cap_t, cap_b], dtype=float)

    soft = [_clip(np.interp(d, pl.s, pl.v_soft) / V_SCALE, 0.0, 1.5) for d in PREVIEW_D]
    gap_soft = _clip((pl.v_soft[0] - v) / 10.0, -2.0, 2.0)
    gap_hard = _clip((pl.v_hard[0] - v) / 10.0, -2.0, 2.0)
    a_soft = 0.7 * vehicle.max_decel
    a_hard = 0.9 * vehicle.max_decel
    slack_soft = _clip(_slack(pl.s, pl.v_soft, v, a_soft) / 40.0, -1.5, 1.5)
    slack_hard = _clip(_slack(pl.s, pl.v_hard, v, a_hard) / 40.0, -1.5, 1.5)
    curv = [_clip(np.interp(d, pl.s, pl.kappa) * 20.0, -3.0, 3.0) for d in CURV_D]
    road = obs.road
    if math.isfinite(pl.obstacle_dist):
        obstacle = [_clip(pl.obstacle_dist / 40.0, -0.1, 1.2), 1.0, _clip(pl.obstacle_shift / 2.0, -1.5, 1.5)]
    else:
        obstacle = [1.2, 0.0, 0.0]
    fd = float(road.finish_distance)
    finish = [_clip(fd / 40.0, 0.0, 1.5), 1.0] if math.isfinite(fd) else [1.5, 0.0]
    vals = (base + soft + [gap_soft, gap_hard, slack_soft, slack_hard] + curv
            + [_clip(road.lateral_offset / 3.0, -2.0, 2.0), _clip(road.heading_error / 0.5, -2.0, 2.0)]
            + obstacle + finish + [cap_t, cap_b])
    out = np.asarray(vals, dtype=float)
    if out.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature length {out.shape[0]} != {FEATURE_DIM}")
    return out

"""Fixed-length policy inputs from ``Observation.road`` and the planner output.

The first 32 names match ``rl_longitudinal_67`` so that policy can be copied into the pedal
head. Three drift inputs are appended: yaw rate, rear-axle slip angle, and the current steer
command. Do not reorder without retraining.
"""
from __future__ import annotations

import math

import numpy as np

from lr_planner import G, Plan, PlannerParams, pedal_caps

PREVIEW_D = (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0, 40.0)
CURV_D = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0)
V_SCALE = 30.0
DRIFT_FEATURES = ("yaw_rate", "rear_slip", "steer")

FEATURE_NAMES = (
    ("speed", "ax", "ay", "throttle", "brake")
    + tuple(f"vsoft_{int(d)}" for d in PREVIEW_D)
    + ("gap_soft", "gap_hard", "slack_soft", "slack_hard")
    + tuple(f"kappa_{int(d)}" for d in CURV_D)
    + ("lateral_offset", "heading_error")
    + ("obstacle_dist", "obstacle_present", "obstacle_shift")
    + ("finish_dist", "finish_present")
    + ("cap_throttle", "cap_brake")
    + DRIFT_FEATURES
)
FEATURE_DIM = len(FEATURE_NAMES)
BASE_FEATURE_DIM = FEATURE_DIM - len(DRIFT_FEATURES)


def _clip(x, lo, hi) -> float:
    return float(np.clip(x, lo, hi))


def rear_slip_deg(vy: float, yaw_rate: float, vx: float, cg_to_rear: float) -> float:
    """Rear-axle slip angle in degrees. Zero below 1 m/s, where the angle is noise."""
    speed = abs(float(vx))
    if speed < 1.0:
        return 0.0
    return math.degrees(math.atan2(float(vy) - cg_to_rear * float(yaw_rate), speed))


def _slack(s, v_ceiling, v, a_dec) -> float:
    """Distance spare before braking at ``a_dec`` is needed to meet the ceiling everywhere ahead."""
    need = np.maximum(v * v - v_ceiling * v_ceiling, 0.0) / (2.0 * max(a_dec, 0.5))
    return float(np.min(s - need))


def _drift_tail(obs, vehicle) -> list[float]:
    slip = rear_slip_deg(obs.vy, obs.yaw_rate, obs.vx, vehicle.cg_to_rear)
    return [
        _clip(obs.yaw_rate / 1.5, -2.0, 2.0),
        _clip(slip / 20.0, -2.0, 2.0),
        _clip(obs.steer, -1.0, 1.0),
    ]


def featurize(obs, vehicle, pl: Plan, prm: PlannerParams | None = None) -> np.ndarray:
    prm = prm or PlannerParams()
    mu_g = max(vehicle.mu * G, 1e-3)
    v = max(float(obs.speed), 0.0)
    cap_t, cap_b = pedal_caps(obs, vehicle, prm)
    base = [
        _clip(v / V_SCALE, 0.0, 1.5), _clip(obs.ax / 10.0, -1.5, 1.5), _clip(obs.ay / mu_g, -1.5, 1.5),
        _clip(obs.throttle, 0.0, 1.0), _clip(obs.brake, 0.0, 1.0),
    ]
    tail = _drift_tail(obs, vehicle)
    if not pl.valid:
        zeros = FEATURE_DIM - len(base) - 2 - len(tail)
        return np.asarray(base + [0.0] * zeros + [cap_t, cap_b] + tail, dtype=float)

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
            + obstacle + finish + [cap_t, cap_b] + tail)
    out = np.asarray(vals, dtype=float)
    if out.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature length {out.shape[0]} != {FEATURE_DIM}")
    return out

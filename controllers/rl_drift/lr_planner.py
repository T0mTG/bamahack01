"""Path planner, speed profiles, safety shield and the analytic "teacher" pedal.

Copied from ``rl_longitudinal_67`` with two drift changes: the hard speed ceiling sits at
the full friction limit, and the forced brake waits until the car is clearly above it.
Steering is still pure pursuit plus a small residual (``combine_steer``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class PlannerParams:
    # soft profile (what the teacher follows). Conservative enough to finish clean;
    # PPO is allowed to push closer to the hard profile.
    lat_frac_soft: float = 0.70
    dec_frac_soft: float = 0.60
    v_horizon_soft: float = 7.0
    # hard profile (shield). 1.0 is the steady-state friction limit, a little above the
    # 0.92 used by rl_longitudinal_67, so a slide can be faster than that profile.
    lat_frac_hard: float = 1.0
    dec_frac_hard: float = 0.82
    v_horizon_hard: float = 8.5
    # obstacle avoidance
    car_half_width: float = 1.08
    obstacle_margin: float = 0.28
    boundary_margin: float = 0.18
    plateau_before: float = 5.0
    plateau_after: float = 6.0
    ramp_base: float = 10.0
    ramp_per_sqrt_m: float = 10.0
    min_ramp: float = 10.0
    shift_speed_cap: float = 11.0
    # pure pursuit
    lookahead_gain: float = 0.5
    lookahead_min: float = 4.0
    # shield
    min_throttle_cap: float = 0.1
    min_brake_cap: float = 0.3
    shield_overspeed: float = 0.8
    # teacher and residuals
    teacher_kp_throttle: float = 0.45
    teacher_kp_brake: float = 0.55
    residual_scale: float = 0.35
    steer_residual_scale: float = 0.40


@dataclass
class Plan:
    s: np.ndarray               # (K,) metres ahead along the course
    path: np.ndarray            # (K,2) path to drive, vehicle frame
    shift: np.ndarray           # (K,) lateral shift of the path vs the centerline (+ left)
    kappa: np.ndarray           # (K,) curvature of the path
    v_soft: np.ndarray          # (K,) comfortable speed ceiling at each preview point
    v_hard: np.ndarray          # (K,) grip-limit speed ceiling at each preview point
    steer: float                # pure pursuit command in [-1, 1]
    obstacle_dist: float        # distance to next obstacle ahead, nan if none
    obstacle_shift: float       # target lateral shift for it
    valid: bool = True
    extra: dict = field(default_factory=dict)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def obstacle_targets(road, prm: PlannerParams):
    """(distance, target_lateral_offset, tight) for each visible obstacle that blocks the centerline."""
    out = []
    s = road.preview_s
    for ob in road.obstacles:
        d = float(ob.distance)
        if d < -3.0 or d > s[-1]:
            continue
        wl = float(np.interp(d, s, road.preview_width_left))
        wr = float(np.interp(d, s, road.preview_width_right))
        pad = prm.car_half_width + prm.obstacle_margin
        lo, hi = float(ob.lateral_min) - pad, float(ob.lateral_max) + pad
        if not (lo < 0.0 < hi):
            continue
        left_max = wl - prm.car_half_width - prm.boundary_margin
        right_min = -(wr - prm.car_half_width - prm.boundary_margin)
        options = []
        if hi <= left_max:
            options.append(hi)
        if lo >= right_min:
            options.append(lo)
        tight = False
        if options:
            target = min(options, key=abs)
        else:
            tight = True
            # no gap wide enough: take the side with the larger clearance, as far out as the boundary allows
            left_room = left_max - (hi - 2 * pad)
            right_room = (lo + 2 * pad) - right_min
            target = left_max if left_room >= right_room else right_min
        out.append((d, float(target), tight))
    return out


def lateral_shift(road, speed: float, prm: PlannerParams):
    """Smooth lateral offset profile (one value per preview point) that steers around obstacles."""
    s = road.preview_s
    shift = np.zeros_like(s)
    targets = obstacle_targets(road, prm)
    if not targets:
        return shift, targets
    pos = np.zeros_like(s)
    neg = np.zeros_like(s)
    for d, c, _ in targets:
        # ramp length depends only on the shift size, not on speed: a speed-dependent ramp changes
        # shape while the car brakes and makes the path jump. The speed profile slows the car for it.
        start = d - prm.plateau_before
        ramp = prm.ramp_base + prm.ramp_per_sqrt_m * math.sqrt(abs(c))
        ramp = float(np.clip(ramp, prm.min_ramp, max(start, prm.min_ramp)))
        f = np.where(
            s < start, _smoothstep((s - (start - ramp)) / ramp),
            np.where(s <= d + prm.plateau_after, 1.0,
                     _smoothstep(((d + prm.plateau_after + ramp) - s) / ramp))) * c
        pos = np.maximum(pos, f)
        neg = np.minimum(neg, f)
    shift = pos + neg
    lo = -(road.preview_width_right - prm.car_half_width - prm.boundary_margin)
    hi = road.preview_width_left - prm.car_half_width - prm.boundary_margin
    return np.clip(shift, lo, np.maximum(hi, lo)), targets


def speed_profile(s, kappa, vehicle, lat_frac, dec_frac, v_horizon, v_cap=None, finish_distance=float("nan")):
    """Backward pass: the fastest speed at each previewed point from which all points ahead are takeable.

    A point's own limit is sqrt(a_lat / |kappa|). Braking is limited by a friction circle (the more
    lateral grip a corner uses, the less is left for braking). The last previewed point is treated as
    a possible hidden slow corner (``v_horizon``) unless the finish line is inside the preview.
    """
    mu_g = vehicle.mu * G
    a_lat = lat_frac * mu_g
    a_dec = dec_frac * vehicle.max_decel
    n = len(s)
    lim = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-4)), vehicle.max_speed)
    if v_cap is not None:
        lim = np.minimum(lim, v_cap)
    if math.isfinite(finish_distance):
        beyond = s > finish_distance + 1.0
        lim = np.where(beyond, vehicle.max_speed, lim)
    else:
        lim = lim.copy()
        lim[-1] = min(lim[-1], v_horizon)
    v = lim.copy()
    for i in range(n - 2, -1, -1):
        ds = s[i + 1] - s[i]
        use = min(v[i + 1] ** 2 * abs(kappa[i]) / mu_g, 0.95)
        dec = a_dec * math.sqrt(1.0 - use * use)
        v[i] = min(lim[i], math.sqrt(v[i + 1] ** 2 + 2.0 * max(dec, 0.5) * ds))
    return v


def plan(obs, vehicle, prm: PlannerParams | None = None) -> Plan:
    """Plan the path and both speed profiles from ``obs.road``."""
    prm = prm or PlannerParams()
    road = obs.road
    if road is None or not road.valid or len(road.preview_s) < 3:
        z = np.zeros(2)
        return Plan(z, np.zeros((2, 2)), z, z, z, z, 0.0, float("nan"), 0.0, valid=False)
    s = np.asarray(road.preview_s, dtype=float)
    cl = np.asarray(road.preview_centerline, dtype=float)
    kappa_c = np.asarray(road.preview_curvature, dtype=float)
    speed = max(float(obs.speed), 0.0)

    shift, targets = lateral_shift(road, speed, prm)
    if targets:
        tang = np.gradient(cl, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack((-tang[:, 1], tang[:, 0]))
        path = cl + normal * shift[:, None]
        d1 = np.gradient(shift, s)
        d2 = np.gradient(d1, s)
        kappa = kappa_c / np.maximum(1.0 - kappa_c * shift, 0.2) + d2
    else:
        path, kappa = cl, kappa_c

    v_cap = np.full_like(s, vehicle.max_speed)
    if targets:
        moving = np.abs(shift) > 0.15
        v_cap = np.where(moving, np.minimum(v_cap, prm.shift_speed_cap), v_cap)
        for d, _, tight in targets:
            if tight:
                v_cap = np.where(np.abs(s - d) < 8.0, np.minimum(v_cap, 6.0), v_cap)
    fd = float(road.finish_distance)
    v_soft = speed_profile(s, kappa, vehicle, prm.lat_frac_soft, prm.dec_frac_soft, prm.v_horizon_soft, v_cap, fd)
    v_hard = speed_profile(s, kappa, vehicle, prm.lat_frac_hard, prm.dec_frac_hard, prm.v_horizon_hard, v_cap, fd)
    steer = pursuit_steer(s, path, speed, vehicle, prm)
    ahead = [t for t in targets if t[0] > -1.0]
    od, osh = (ahead[0][0], ahead[0][1]) if ahead else (float("nan"), 0.0)
    return Plan(s, path, shift, kappa, v_soft, v_hard, steer, od, osh)


def pursuit_steer(s, path, speed, vehicle, prm: PlannerParams) -> float:
    lookahead = max(prm.lookahead_min, prm.lookahead_gain * speed + 2.0)
    tx = float(np.interp(lookahead, s, path[:, 0]))
    ty = float(np.interp(lookahead, s, path[:, 1]))
    alpha = math.atan2(ty, max(tx, 0.1))
    delta = math.atan2(2.0 * vehicle.wheelbase * math.sin(alpha), max(math.hypot(tx, ty), 1.0))
    return float(np.clip(delta / vehicle.max_steer, -1.0, 1.0))


def pedal_caps(obs, vehicle, prm: PlannerParams) -> tuple[float, float]:
    """Throttle and brake ceilings from how much grip cornering is already using."""
    lat_use = min(abs(obs.ay) / max(vehicle.mu * G, 1e-3), 1.0)
    return (max(1.0 - lat_use ** 2, prm.min_throttle_cap),
            max(math.sqrt(1.0 - lat_use ** 2), prm.min_brake_cap))


def u_to_pedals(u: float) -> tuple[float, float]:
    u = float(np.clip(u, -1.0, 1.0))
    return (u, 0.0) if u >= 0.0 else (0.0, -u)


def shield(u: float, obs, pl: Plan, vehicle, prm: PlannerParams | None = None) -> tuple[float, float]:
    """Turn the policy's pedal request into a safe (throttle, brake). Never both pedals."""
    prm = prm or PlannerParams()
    if not pl.valid or not math.isfinite(u):
        return 0.0, 0.6
    throttle, brake = u_to_pedals(u)
    v = max(float(obs.speed), 0.0)
    over = v - float(pl.v_hard[0])
    if over > prm.shield_overspeed:
        return 0.0, float(np.clip(0.4 + 0.3 * over, 0.4, 1.0))
    cap_t, cap_b = pedal_caps(obs, vehicle, prm)
    return min(throttle, cap_t), min(brake, cap_b)


def teacher_u(obs, pl: Plan, prm: PlannerParams | None = None) -> float:
    """Pedal that tracks v_soft. Used as a no-policy fallback and as the residual baseline."""
    prm = prm or PlannerParams()
    if not pl.valid:
        return -0.6
    v = max(float(obs.speed), 0.0)
    err = float(pl.v_soft[min(1, len(pl.v_soft) - 1)]) - v
    return float(np.clip(prm.teacher_kp_throttle * err if err >= 0 else prm.teacher_kp_brake * err, -1.0, 1.0))


def combine_u(teacher: float, residual: float, prm: PlannerParams | None = None) -> float:
    """``u = teacher + scale * residual``. A zero residual is exactly the analytic controller."""
    prm = prm or PlannerParams()
    return float(np.clip(teacher + prm.residual_scale * residual, -1.0, 1.0))


def combine_steer(pursuit: float, residual: float, prm: PlannerParams | None = None) -> float:
    """``steer = clip(pursuit + scale * residual)``. A zero residual is pure pursuit."""
    prm = prm or PlannerParams()
    return float(np.clip(float(pursuit) + prm.steer_residual_scale * float(residual), -1.0, 1.0))

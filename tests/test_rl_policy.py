"""Planner, features, self-contained controller, track styles, and a few bicycle steps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from autocross.api import Observation, RoadModel
from autocross.track import load
from rl.env import NOMINAL, LongitudinalEnv
from lr_features import FEATURE_DIM, FEATURE_NAMES, featurize
from lr_net import MLPPolicy
from lr_planner import PlannerParams, obstacle_targets, plan, shield, speed_profile, teacher_u, u_to_pedals

ROOT = Path(__file__).resolve().parent.parent
CTRL_DIR = ROOT / "controllers" / "rl_longitudinal"


def test_feature_names_match_dimension():
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_pedals_are_exclusive():
    assert u_to_pedals(0.4) == (0.4, 0.0)
    assert u_to_pedals(-0.3) == (0.0, 0.3)
    assert u_to_pedals(2.0) == (1.0, 0.0)


def test_speed_profile_is_braking_feasible():
    s = np.linspace(0.0, 40.0, 41)
    kappa = np.zeros_like(s)
    kappa[30:] = 1.0 / 12.0
    class V:
        mu, max_decel, max_speed = 0.95, 9.0, 40.0
    v = speed_profile(s, kappa, V(), 0.7, 0.6, 8.0)
    a_dec = 0.6 * 9.0
    for i in range(len(s) - 1):
        ds = s[i + 1] - s[i]
        assert v[i] ** 2 <= v[i + 1] ** 2 + 2.0 * a_dec * ds + 1e-4


def test_planner_shifts_around_a_blocking_obstacle():
    s = np.linspace(0.0, 40.0, 41)
    cl = np.column_stack((s, np.zeros_like(s)))
    n = np.column_stack((np.zeros_like(s), np.ones_like(s)))
    road = RoadModel(
        valid=True, range_m=40.0, lateral_offset=0.0, heading_error=0.0, curvature=0.0,
        width_left=2.75, width_right=2.75, dist_to_left_boundary=2.75, dist_to_right_boundary=2.75,
        preview_s=s, preview_curvature=np.zeros_like(s), preview_centerline=cl,
        preview_width_left=np.full_like(s, 2.75), preview_width_right=np.full_like(s, 2.75),
        preview_left_boundary=cl + n * 2.75, preview_right_boundary=cl - n * 2.75,
        corners=[], obstacles=[], finish_distance=float("nan"))
    from autocross.api import ObstacleInfo
    road.obstacles = [ObstacleInfo(20.0, -0.4, 1.6, np.array([20.0, 0.6]), np.zeros((1, 2)))]
    obs = Observation(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8.0, 0, 0, 0, 0, 0, True, road=road)
    class V:
        mu, max_decel, max_accel, max_speed, wheelbase, max_steer = 0.95, 9.0, 5.2, 40.0, 3.0, 0.61
    pl = plan(obs, V())
    assert pl.valid
    at = float(np.interp(20.0, pl.s, pl.shift))
    assert at < -0.2
    assert (np.abs(pl.shift) <= 2.75 - 1.08).all()
    targets = obstacle_targets(road, PlannerParams())
    assert targets and targets[0][1] < 0.0


def test_shield_forces_brake_when_too_fast():
    s = np.linspace(0.0, 40.0, 41)
    from lr_planner import Plan
    pl = Plan(s, np.column_stack((s, np.zeros_like(s))), np.zeros_like(s), np.zeros_like(s),
              np.full_like(s, 8.0), np.full_like(s, 8.0), 0.0, float("nan"), 0.0)
    class V:
        mu, max_decel, max_speed = 0.95, 9.0, 40.0
    obs = Observation(0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 12.0, 0, 0, 0, 0, 0, True)
    th, br = shield(1.0, obs, pl, V())
    assert th == 0.0 and br > 0.3


def test_policy_npz_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    policy = MLPPolicy.random(FEATURE_DIM, (32, 32), rng)
    obs = rng.normal(size=FEATURE_DIM)
    before = policy.act(obs)
    path = tmp_path / "policy.npz"
    policy.save(path)
    after = MLPPolicy.load(path).act(obs)
    assert after == pytest.approx(before)


def test_env_reset_and_steps_stay_finite():
    env = LongitudinalEnv(randomize=False, max_time_s=20.0, seed=0)
    features = env.reset(track=ROOT / "tracks" / "practice_basic.yaml", seed=1, rand=NOMINAL)
    assert features.shape == (FEATURE_DIM,)
    assert np.isfinite(features).all()
    total = 0.0
    info = {}
    for _ in range(20):
        features, reward, done, info = env.step(0.4)
        total += reward
        assert features.shape == (FEATURE_DIM,)
        assert np.isfinite(features).all()
        assert np.isfinite(reward)
        if done:
            break
    assert info["progress"] > 0.0
    assert np.allclose(featurize(env._obs, env.vehicle, env._plan), features)


def test_controller_folder_imports_without_rl(tmp_path):
    """The submission folder must load when `rl` and `torch` are not importable."""
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "rl.py").write_text("raise ImportError('rl hidden')\n")
    (hidden / "torch.py").write_text("raise ImportError('torch hidden')\n")
    from autocross.sandbox import load_controller_class
    old = list(sys.path)
    try:
        sys.path.insert(0, str(hidden))
        sys.modules.pop("rl", None)
        sys.modules.pop("torch", None)
        cls = load_controller_class(str(CTRL_DIR / "controller.py"))
        env = LongitudinalEnv(randomize=False, max_time_s=10.0, seed=0)
        env.reset(track=ROOT / "tracks" / "brake_hairpin.yaml", seed=0, rand=NOMINAL)
        policy = MLPPolicy.random(FEATURE_DIM, rng=np.random.default_rng(3))
        path = tmp_path / "policy.npz"
        policy.save(path)
        controller = cls({"policy": str(path)})
        controller.on_start(env.track.info(False, 20.0), env.vehicle)
        command = controller.step(env._obs)
        assert -1.0 <= command.steer <= 1.0
        assert command.throttle == 0.0 or command.brake == 0.0
        assert command.path is not None
        assert command.path_frame == "vehicle"
    finally:
        sys.path[:] = old


def test_teacher_mode_drives():
    from autocross.sandbox import load_controller_class
    env = LongitudinalEnv(randomize=False, max_time_s=10.0, seed=0)
    env.reset(track=ROOT / "tracks" / "brake_hairpin.yaml", seed=0, rand=NOMINAL)
    cls = load_controller_class(str(CTRL_DIR / "controller.py"))
    controller = cls({"mode": "teacher"})
    controller.on_start(env.track.info(False, 20.0), env.vehicle)
    command = controller.step(env._obs)
    assert command.throttle == 0.0 or command.brake == 0.0
    assert teacher_u(env._obs, env._plan) == pytest.approx(command.debug["u"], abs=1e-6)


@pytest.mark.parametrize("rel", [
    "tracks/brake_hairpin.yaml",
    "tracks/accel_straights.yaml",
    "tracks/generated/open_fast.yaml",
    "tracks/generated/open_obstacles.yaml",
    "tracks/generated/open_technical.yaml",
    "tracks/generated/closed_loop.yaml",
])
def test_existing_courses_validate(rel):
    track = load(ROOT / rel)
    assert track.validate() == []
    assert track.length > 100.0


@pytest.mark.parametrize("style", ["braking", "accel", "technical", "fast"])
def test_track_styles_validate(style):
    from autocross import trackgen
    from autocross.track import build
    spec = trackgen.generate(11 + hash(style) % 50, style)
    track = build(spec)
    assert track.validate() == []
    assert track.length > 80.0

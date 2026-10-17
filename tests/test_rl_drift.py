"""Drift controller: extra features line up with the 67 policy, and zero steer is pure pursuit."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIFT = ROOT / "controllers" / "rl_drift"
BASE67 = ROOT / "controllers" / "rl_longitudinal_67"
_LR = ("lr_features", "lr_planner", "lr_net")


@pytest.fixture
def drift():
    saved_mod = {n: sys.modules.get(n) for n in _LR}
    saved_path = list(sys.path)
    for n in _LR:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(DRIFT))
    mods = {n: importlib.import_module(n) for n in _LR}
    try:
        yield mods
    finally:
        sys.path[:] = saved_path
        for n in _LR:
            sys.modules.pop(n, None)
            if saved_mod[n] is not None:
                sys.modules[n] = saved_mod[n]


def _load_names(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FEATURE_NAMES


def test_features_extend_the_67_policy(drift):
    feat = drift["lr_features"]
    assert feat.FEATURE_DIM == 35
    assert feat.BASE_FEATURE_DIM == 32
    assert feat.FEATURE_NAMES[-3:] == ("yaw_rate", "rear_slip", "steer")
    saved = {n: sys.modules.pop(n) for n in _LR}
    sys.path.insert(0, str(BASE67))
    try:
        base_names = _load_names(BASE67 / "lr_features.py", "feat67_names_for_drift_test")
    finally:
        sys.path.pop(0)
        for n, mod in saved.items():
            sys.modules[n] = mod
    assert feat.FEATURE_NAMES[:32] == base_names


def test_zero_steer_residual_is_pure_pursuit(drift):
    planner = drift["lr_planner"]
    prm = planner.PlannerParams()
    assert prm.lat_frac_hard == 1.0
    assert prm.shield_overspeed == 0.8
    assert prm.steer_residual_scale == pytest.approx(0.40)
    assert planner.combine_steer(0.25, 0.0, prm) == pytest.approx(0.25)
    assert planner.combine_steer(0.25, 1.0, prm) == pytest.approx(0.65)
    assert planner.combine_steer(0.9, 1.0, prm) == pytest.approx(1.0)


def test_shield_waits_until_clearly_over_the_limit(drift):
    planner = drift["lr_planner"]
    s = np.linspace(0.0, 40.0, 41)
    pl = planner.Plan(s, np.column_stack((s, np.zeros_like(s))), np.zeros_like(s), np.zeros_like(s),
                      np.full_like(s, 8.0), np.full_like(s, 8.0), 0.2, float("nan"), 0.0)

    class V:
        mu, max_decel, max_speed = 0.95, 9.0, 40.0

    from autocross.api import Observation
    mild = Observation(0, 0, 0, 0, 0, 8.5, 0, 0, 0, 0, 8.5, 0, 0, 0, 0, 0, True)
    th, br = planner.shield(0.5, mild, pl, V())
    assert th == pytest.approx(0.5)
    assert br == 0.0
    fast = Observation(0, 0, 0, 0, 0, 9.2, 0, 0, 0, 0, 9.2, 0, 0, 0, 0, 0, True)
    th, br = planner.shield(0.5, fast, pl, V())
    assert th == 0.0 and br > 0.3


def test_warm_start_keeps_the_67_pedal_and_zeros_steer(drift):
    from rl.train_drift import SOURCE_67, expand_from_67
    old = drift["lr_net"].MLPPolicy.load(SOURCE_67)
    new = expand_from_67(SOURCE_67)
    rng = np.random.default_rng(0)
    x32 = rng.normal(size=32)
    new_y = new.act(np.concatenate([x32, np.zeros(3)]))
    assert new.act_dim == 2
    assert new_y[0] == pytest.approx(float(np.asarray(old.act(x32)).reshape(-1)[0]), abs=1e-6)
    assert new_y[1] == pytest.approx(0.0, abs=1e-6)


def test_controller_with_zero_steer_head_matches_pursuit(drift, tmp_path):
    from autocross.sandbox import load_controller_class
    from rl.env import NOMINAL, LongitudinalEnv

    feat = drift["lr_features"]
    net = drift["lr_net"]
    planner = drift["lr_planner"]
    env = LongitudinalEnv(randomize=False, max_time_s=15.0, seed=0)
    env.reset(track=ROOT / "tracks" / "brake_hairpin.yaml", seed=0, rand=NOMINAL)
    policy = net.MLPPolicy.random(feat.FEATURE_DIM, (32, 32), act_dim=2, rng=np.random.default_rng(1))
    policy.weights[-1][:] = 0.0
    policy.biases[-1][:] = 0.0
    assert np.allclose(policy.act(np.zeros(feat.FEATURE_DIM)), 0.0)
    path = tmp_path / "policy.npz"
    policy.save(path)
    cls = load_controller_class(str(DRIFT / "controller.py"))
    controller = cls({"policy": str(path)})
    controller.on_start(env.track.info(False, 20.0), env.vehicle)
    command = controller.step(env._obs)
    pl = planner.plan(env._obs, env.vehicle, controller.prm)
    assert command.steer == pytest.approx(pl.steer)
    assert command.throttle == 0.0 or command.brake == 0.0
    features = feat.featurize(env._obs, env.vehicle, pl, controller.prm)
    assert features.shape == (35,)
    assert np.isfinite(features).all()

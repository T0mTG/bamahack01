"""Drift RL controller (self-contained folder).

Steering: pure pursuit plus a small residual from the network, so the path still leads and the
policy can rotate the car. Throttle and brake: the analytic teacher plus a pedal residual,
wrapped by the same style of safety shield as ``rl_longitudinal_67``. The shield brakes a
little later, which is the room a slide needs.

Only numpy and ``autocross.api`` are used at run time. No torch, no GPU, no files outside
this folder.

Train with ``python -m rl.train_drift`` (writes ``policy.npz`` here).

Config keys (``--controller-config cfg.yaml``):
    mode: policy | teacher     teacher = analytic speed controller, no network (default: policy)
    policy: path               alternative weights file
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams  # noqa: E402
from lr_features import featurize  # noqa: E402
from lr_net import MLPPolicy  # noqa: E402
from lr_planner import PlannerParams, combine_steer, combine_u, plan, shield, teacher_u  # noqa: E402

POLICY_FILE = _HERE / "policy.npz"


class DriftRL(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        self.prm = PlannerParams()
        self.mode = str(self.config.get("mode", "policy"))
        self.policy = None
        if self.mode == "policy":
            path = Path(self.config.get("policy", POLICY_FILE))
            if path.is_file():
                self.policy = MLPPolicy.load(path)
            else:
                self.mode = "teacher"

    def step(self, obs: Observation) -> Command:
        pl = plan(obs, self.vehicle, self.prm)
        if not pl.valid:
            return Command(brake=1.0)
        u_t = teacher_u(obs, pl, self.prm)
        pedal_r, steer_r = 0.0, 0.0
        if self.policy is not None:
            raw = np.atleast_1d(self.policy.act(featurize(obs, self.vehicle, pl, self.prm)))
            pedal_r = float(raw[0])
            if raw.size > 1:
                steer_r = float(raw[1])
        u = combine_u(u_t, pedal_r, self.prm)
        steer = combine_steer(pl.steer, steer_r, self.prm)
        throttle, brake = shield(u, obs, pl, self.vehicle, self.prm)
        return Command(
            steer=steer, throttle=throttle, brake=brake,
            path=pl.path, path_frame="vehicle", target_speed=float(pl.v_soft[min(1, len(pl.v_soft) - 1)]),
            debug={"u": float(u), "steer_residual": float(steer_r),
                   "v_soft": float(pl.v_soft[0]), "v_hard": float(pl.v_hard[0]),
                   "shift": float(pl.shift[0])},
        )

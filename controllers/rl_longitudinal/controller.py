"""Longitudinal RL controller (self-contained submission folder).

Steering: pure pursuit on a path that swings around obstacles (``lr_planner``).
Throttle and brake: a small numpy network (``lr_net``, weights in ``policy.npz``) wrapped by a
safety shield. Only numpy and ``autocross.api`` are used at run time. No torch, no GPU, no
files outside this folder.

Copy this whole folder to ``teams/<name>/`` to submit it. Train or replace the weights with
``python -m rl.train`` (see docs/HOW_TO_RUN.md).

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
from lr_planner import PlannerParams, combine_u, plan, shield, teacher_u  # noqa: E402

POLICY_FILE = _HERE / "policy.npz"


class LongitudinalRL(Controller):
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
                self.mode = "teacher"   # no weights yet: still drives, with the analytic controller

    def step(self, obs: Observation) -> Command:
        pl = plan(obs, self.vehicle, self.prm)
        if not pl.valid:
            return Command(brake=1.0)
        u_t = teacher_u(obs, pl, self.prm)
        if self.policy is not None:
            u = combine_u(u_t, self.policy.act(featurize(obs, self.vehicle, pl, self.prm)), self.prm)
        else:
            u = u_t
        throttle, brake = shield(u, obs, pl, self.vehicle, self.prm)
        return Command(
            steer=pl.steer, throttle=throttle, brake=brake,
            path=pl.path, path_frame="vehicle", target_speed=float(pl.v_soft[min(1, len(pl.v_soft) - 1)]),
            debug={"u": float(u), "v_soft": float(pl.v_soft[0]), "v_hard": float(pl.v_hard[0]),
                   "shift": float(pl.shift[0])},
        )

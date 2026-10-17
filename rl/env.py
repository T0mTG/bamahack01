"""Bicycle-model rollouts for the speed policy.

One environment = one course at a time, 40 Hz physics and 20 Hz decisions, the same road model,
actuator model (steering slew limit, optional delay) and rules monitor as a real run. Steering
comes from the planner; the action is one pedal number ``u`` in [-1, 1] passed through the same
safety shield the controller uses.

Domain randomization (``randomize=True``) varies grip, mass, brake/drive force, actuation delay,
road-model noise and sight distance every episode, because half of the official score is CARLA,
whose dynamics are only approximated by the bicycle model.

Reward per control step (all scaled so that one point of the real score is about one unit):
    +0.05 / m of progress, -0.005 time cost
    -0.08 x fraction of the step with rear slip > 6 deg, -0.05 x fraction with grip demand > 110 %
    -0.03 |du| (pedal smoothness)
    -3 cone hit, -6 off course, -8 missed gate, -40 DNF
    finish: +30 x clip((2 - time/reference) / 1.05, 0, 1)   (the completion-time score, 30 points)
"""
from __future__ import annotations

import copy
import math
from collections import deque
from pathlib import Path

import numpy as np
import rl  # noqa: F401  (puts the controller folder on sys.path)
from autocross.api import Observation, VehicleParams
from autocross.backends.bicycle import BicycleBackend
from autocross.config import load_sim_config
from autocross.monitor import RaceMonitor
from autocross.perception import RoadModelSensor
from autocross.track import build as build_track
from autocross.track import load as load_track
from lr_features import FEATURE_DIM, featurize
from lr_planner import G, PlannerParams, plan, shield, teacher_u

NOMINAL = {"friction_scale": 1.0, "mass_scale": 1.0, "drive_scale": 1.0, "brake_scale": 1.0, "delay": 0,
           "offset_noise": 0.0, "heading_noise_deg": 0.0, "curvature_noise": 0.0, "range_m": 40.0}


def sample_randomization(rng: np.random.Generator, strength: float = 1.0) -> dict:
    """Per-episode physics and perception perturbation (``strength`` 0 = nominal)."""
    s = float(strength)
    noisy = rng.random() < 0.4
    return {
        "friction_scale": 1.0 + s * float(rng.uniform(-0.2, 0.05)),
        "mass_scale": 1.0 + s * float(rng.uniform(-0.1, 0.2)),
        "drive_scale": 1.0 + s * float(rng.uniform(-0.15, 0.15)),
        "brake_scale": 1.0 + s * float(rng.uniform(-0.15, 0.15)),
        "delay": int(rng.choice([0, 0, 0, 1, 2])) if s > 0 else 0,
        "offset_noise": float(rng.uniform(0.0, 0.12)) * s if noisy else 0.0,
        "heading_noise_deg": float(rng.uniform(0.0, 0.6)) * s if noisy else 0.0,
        "curvature_noise": float(rng.uniform(0.0, 0.002)) * s if noisy else 0.0,
        "range_m": float(rng.choice([40.0, 40.0, 30.0, 25.0])) if s > 0 else 40.0,
    }


class LongitudinalEnv:
    def __init__(self, cfg: dict | None = None, randomize: bool = True, max_time_s: float = 150.0, seed: int = 0,
                 prm: PlannerParams | None = None):
        self.base_cfg = cfg or load_sim_config()
        self.randomize = randomize
        self.max_time_s = float(max_time_s)
        self.rng = np.random.default_rng(seed)
        self.prm = prm or PlannerParams()
        sim = self.base_cfg["sim"]
        self.dt = float(sim["physics_dt"])
        self.steps_per_control = int(round(1.0 / (float(sim["control_hz"]) * self.dt)))
        self.control_dt = self.steps_per_control * self.dt
        self._tracks: dict = {}
        self.track = None
        self.reference_time = 1.0
        self.rand = dict(NOMINAL)
        self.vehicle: VehicleParams | None = None
        self.obs_dim = FEATURE_DIM

    # ------------------------------------------------------------------ setup
    def _get_track(self, source):
        if isinstance(source, dict):
            key = source.get("name", "") + str(len(source.get("segments", source.get("centerline", []))))
            if key not in self._tracks:
                self._tracks[key] = build_track(source)
            return self._tracks[key]
        key = str(source)
        if key not in self._tracks:
            self._tracks[key] = load_track(key)
        return self._tracks[key]

    def _episode_cfg(self) -> dict:
        cfg = copy.deepcopy(self.base_cfg)
        bic = cfg["vehicle"].setdefault("bicycle", {})
        r = self.rand
        if "max_drive_force" in bic:
            bic["max_drive_force"] = float(bic["max_drive_force"]) * r["drive_scale"]
        if "max_brake_force" in bic:
            bic["max_brake_force"] = float(bic["max_brake_force"]) * r["brake_scale"]
        rm = cfg.setdefault("road_model", {})
        rm["offset_noise_std_m"] = r["offset_noise"]
        rm["heading_noise_std_deg"] = r["heading_noise_deg"]
        rm["curvature_noise_std"] = r["curvature_noise"]
        rm["range_m"] = r["range_m"]
        cfg["actuators"]["delay_steps"] = int(r["delay"])
        return cfg

    def reset(self, track=None, seed: int | None = None, rand: dict | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if track is not None:
            self.track = self._get_track(track)
        if rand is not None:
            self.rand = {**NOMINAL, **rand}
        else:
            self.rand = sample_randomization(self.rng) if self.randomize else dict(NOMINAL)
        cfg = self._episode_cfg()
        self.backend = BicycleBackend(cfg["sim"])
        perturb = {"friction_scale": self.rand["friction_scale"], "mass_scale": self.rand["mass_scale"]}
        phys = self.backend.setup(self.track, cfg["vehicle"], perturb, self.rng)
        # the controller is told the published (nominal) limits, not the randomized truth
        v = self.base_cfg["vehicle"]
        self.vehicle = VehicleParams(
            mass=phys.mass, wheelbase=phys.wheelbase, track_width=phys.track_width, length=phys.length,
            width=phys.width, cg_to_front=phys.cg_to_front, cg_to_rear=phys.cg_to_rear, max_steer=phys.max_steer,
            max_steer_rate=float(v["steer_rate_limit"]), mu=float(v["mu"]), max_accel=float(v["max_accel"]),
            max_decel=float(v["max_decel"]), max_speed=float(v["max_speed"]), control_dt=self.control_dt,
            actuation_delay_steps=int(self.rand["delay"]))
        ref = self.track.reference_time
        self.reference_time = float(ref) if ref else self.track.reference_time_estimate(self.vehicle)
        self.steer_rate = self.vehicle.max_steer_rate * self.dt
        x, y, yaw = self.track.spawn_pose()
        self._state = self.backend.reset(x, y, yaw)
        self.monitor = RaceMonitor(self.track, cfg["rules"], self.vehicle, self.max_time_s)
        self.road_sensor = RoadModelSensor(self.track, cfg.get("road_model", {}), self.rng)
        self.delay = deque([(0.0, 0.0, 1.0)] * int(self.rand["delay"]))
        self.app = [0.0, 0.0, 0.0]
        self.step_i = 0
        self._event_i = 0
        self._u_prev = 0.0
        return self._observe()

    # ------------------------------------------------------------------ interaction
    def observation(self) -> Observation:
        st = self._state
        return Observation(
            t=st.t, step=self.step_i, x=st.x, y=st.y, yaw=st.yaw, vx=st.vx, vy=st.vy,
            yaw_rate=st.yaw_rate, ax=st.ax, ay=st.ay, speed=st.speed,
            steer=self.app[0], throttle=self.app[1], brake=self.app[2],
            progress=self.monitor.progress, lap=self.monitor.lap, timing_started=self.monitor.started,
            road=self.road_sensor.measure(st.x, st.y, st.yaw, self.monitor.progress))

    def _observe(self) -> np.ndarray:
        self._obs = self.observation()
        self._plan = plan(self._obs, self.vehicle, self.prm)
        self._feat = featurize(self._obs, self.vehicle, self._plan, self.prm)
        return self._feat

    def teacher_action(self) -> float:
        return teacher_u(self._obs, self._plan, self.prm)

    def step(self, u: float):
        u = float(np.clip(u, -1.0, 1.0))
        obs, pl = self._obs, self._plan
        throttle, brake = shield(u, obs, pl, self.vehicle, self.prm)
        steer = pl.steer if pl.valid else 0.0
        self.delay.append((steer, throttle, brake))
        cmd = self.delay.popleft()
        progress_before = self.monitor.progress
        slip_ticks = grip_ticks = 0
        mu_g = self.vehicle.mu * G
        for _ in range(self.steps_per_control):
            self.app[0] = float(np.clip(
                self.app[0] + np.clip(cmd[0] - self.app[0], -self.steer_rate, self.steer_rate), -1.0, 1.0))
            self.app[1], self.app[2] = cmd[1], cmd[2]
            self.backend.apply(*self.app)
            st = self._state = self.backend.tick()
            self.monitor.update(st, self.dt)
            if st.speed > 3.0:
                vx = max(abs(st.vx), 0.1)
                rear = math.degrees(math.atan2(st.vy - self.vehicle.cg_to_rear * st.yaw_rate, vx))
                slip_ticks += abs(rear) > 6.0
                demand = st.speed ** 2 * abs(math.tan(self.app[0] * self.vehicle.max_steer)) / self.vehicle.wheelbase / mu_g
                grip_ticks += demand > 1.1
            if self.monitor.done:
                break
        self.step_i += 1
        n = float(self.steps_per_control)
        dprog = self.monitor.progress - progress_before
        reward = (0.05 * dprog - 0.005 - 0.08 * slip_ticks / n - 0.05 * grip_ticks / n
                  - 0.03 * abs(u - self._u_prev))
        self._u_prev = u
        info = {"cones": 0, "off_course": 0, "gates": 0}
        for event in self.monitor.events[self._event_i:]:
            if event.kind == "cone_hit":
                reward -= 3.0
                info["cones"] += 1
            elif event.kind == "off_course":
                reward -= 6.0
                info["off_course"] += 1
            elif event.kind == "missed_gate":
                reward -= 8.0
                info["gates"] += 1
            elif event.kind == "dnf":
                reward -= 40.0
            elif event.kind == "finish":
                elapsed = float(event.detail.get("time", self.monitor.elapsed(self._state.t)))
                ratio = elapsed / max(self.reference_time, 1e-3)
                reward += 30.0 * float(np.clip((2.0 - ratio) / 1.05, 0.0, 1.0))
        self._event_i = len(self.monitor.events)
        done = bool(self.monitor.done)
        finished = self.monitor.finish_time is not None and self.monitor.dnf is None
        info.update({
            "progress": float(self.monitor.progress), "speed": float(self._state.speed), "finished": finished,
            "dnf": self.monitor.dnf or "", "track": self.track.name, "slip": slip_ticks / n, "grip": grip_ticks / n,
            "time": float(self.monitor.finish_time - self.monitor.start_time) if finished and self.monitor.start_time is not None else float("nan"),
            "reference": self.reference_time,
        })
        return self._observe(), float(reward), done, info


def run_episode(env: LongitudinalEnv, policy_act, track, seed: int, rand: dict | None = None) -> dict:
    """Drive one full episode with ``policy_act(features, env) -> u`` and return a result row."""
    env.reset(track=track, seed=seed, rand=rand)
    total, cones, off, gates, slip = 0.0, 0, 0, 0, 0.0
    steps, info = 0, {}
    done = False
    while not done:
        u = policy_act(env._feat, env)
        _, r, done, info = env.step(u)
        total += r
        cones += info["cones"]
        off += info["off_course"]
        gates += info["gates"]
        slip += info["slip"]
        steps += 1
    return {"track": info["track"], "return": total, "finished": info["finished"], "dnf": info["dnf"],
            "time": info["time"], "reference": info["reference"], "cones": cones, "off_course": off,
            "gates": gates, "slip_frac": slip / max(steps, 1), "progress": info["progress"]}


def load_pool(directory: str | Path) -> list[dict]:
    """Course specs (YAML) from a directory, for use as ``reset(track=spec)``."""
    import yaml
    specs = []
    for p in sorted(Path(directory).glob("*.yaml")):
        with open(p) as f:
            spec = yaml.safe_load(f)
        spec.setdefault("name", p.stem)
        specs.append(spec)
    return specs

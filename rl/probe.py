"""Drive the analytic teacher (or a trained policy) in the fast in-process env and print one row per course.

    python -m rl.probe                                   # teacher on the public courses
    python -m rl.probe --planner '{"lat_frac_soft": 0.7}' --tracks hackathon_test practice_technical
    python -m rl.probe --policy controllers/rl_longitudinal/policy.npz
    python -m rl.probe --randomize                       # with domain randomization

Used to tune ``PlannerParams`` without the sandbox/CARLA overhead.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rl  # noqa: F401
from lr_net import MLPPolicy
from lr_planner import PlannerParams
from rl.env import NOMINAL, LongitudinalEnv, run_episode

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ["hackathon_test", "practice_technical", "practice_basic", "practice_obstacles", "brake_hairpin",
          "accel_straights", "practice_loop"]


def resolve(name: str) -> str:
    p = Path(name)
    return str(p if p.suffix else ROOT / "tracks" / f"{name}.yaml")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", nargs="*", default=PUBLIC, help="course names under tracks/ or YAML paths")
    ap.add_argument("--planner", default="{}", help="JSON overrides for PlannerParams")
    ap.add_argument("--policy", help="policy.npz to drive (default: analytic teacher)")
    ap.add_argument("--randomize", action="store_true", help="randomize physics/perception per episode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prm = PlannerParams(**json.loads(args.planner))
    env = LongitudinalEnv(randomize=args.randomize, prm=prm, seed=args.seed)
    if args.policy:
        net = MLPPolicy.load(args.policy)
        act = lambda f, e: net.act(f)  # noqa: E731
    else:
        act = lambda f, e: e.teacher_action()  # noqa: E731
    total = 0.0
    for name in args.tracks:
        t0 = time.perf_counter()
        r = run_episode(env, act, resolve(name), seed=args.seed, rand=None if args.randomize else NOMINAL)
        total += r["return"]
        t = f"{r['time']:.1f}" if r["finished"] else "  - "
        print(f"{name:22s} fin={int(r['finished'])} dnf={r['dnf'] or '-':8s} t={t:>6s} ref={r['reference']:.1f} "
              f"cones={r['cones']} off={r['off_course']} gates={r['gates']} slip={r['slip_frac']:.3f} "
              f"ret={r['return']:7.1f} ({time.perf_counter() - t0:.1f}s)")
    print(f"total return {total:.1f}")


if __name__ == "__main__":
    main()

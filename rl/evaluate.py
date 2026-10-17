"""Compare the RL controller to the baseline on public courses, holdout tracks and perturbations.

    python -m rl.evaluate
    python -m rl.evaluate --teacher          # analytic planner, no network
    python -m rl.evaluate --no-baseline      # RL only (faster)

Uses the official `autocross.runner` so the numbers match `python -m autocross run`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from autocross.config import load_sim_config
from autocross.runner import run

ROOT = Path(__file__).resolve().parent.parent
CTRL = ROOT / "controllers" / "rl_longitudinal" / "controller.py"
BASE = ROOT / "controllers" / "baseline_pure_pursuit.py"
PUBLIC = [
    ROOT / "tracks" / "hackathon_test.yaml",
    ROOT / "tracks" / "practice_technical.yaml",
    ROOT / "tracks" / "practice_basic.yaml",
    ROOT / "tracks" / "practice_obstacles.yaml",
    ROOT / "tracks" / "brake_hairpin.yaml",
    ROOT / "tracks" / "accel_straights.yaml",
]
PERTURB = {
    "nominal": {},
    "low_grip": {"friction_scale": 0.8},
    "heavy": {"mass_scale": 1.15},
    "latency": {"actuation_delay_steps": 2},
    "noisy_road": {"road_offset_noise": 0.1, "road_heading_noise_deg": 0.5, "road_curvature_noise": 0.002},
    "short_sight": {"road_range_m": 25},
}


def _row(summary: dict) -> dict:
    m, sc = summary["metrics"], summary["score"]
    c, p = m["completion"], m["penalties"]
    return {
        "track": summary["track"]["name"],
        "team": summary["team"],
        "perturb": summary.get("perturbation") or {},
        "finished": c["finished"],
        "dnf": c["dnf_reason"] or "",
        "time": c["time_s"],
        "ref": c["reference_time_s"],
        "cones": p["cone_hit"],
        "off": p["off_course"],
        "gates": p["missed_gate"],
        "tech": sc["technical_points"],
        "path": m["deviation"]["path_err_mean_m"],
        "slip": m["dynamics"]["rear_slip_max_deg"],
        "out": summary.get("out_dir", ""),
    }


def _print(rows):
    print(f"{'controller':<22} {'track':<22} {'pert':<12} {'fin':>3} {'time':>7} {'ref':>6} "
          f"{'cone':>4} {'off':>3} {'tech':>6} {'slip':>6}")
    for r in rows:
        pert = ",".join(r["perturb"]) if r["perturb"] else "-"
        t = f"{r['time']:.1f}" if r["finished"] else "DNF"
        print(f"{r['team']:<22} {r['track']:<22} {pert:<12} {int(r['finished']):>3} {t:>7} {r['ref']:6.1f} "
              f"{r['cones']:4d} {r['off']:3d} {r['tech']:6.1f} {r['slip']:6.1f}")


def _drive(controller, tracks, team, cfg, perturb=None, controller_cfg=None):
    rows = []
    for tp in tracks:
        if not Path(tp).is_file():
            continue
        s = run(cfg, str(controller), str(tp), team=team, backend="bicycle",
                perturb=perturb or {}, controller_cfg=controller_cfg, verbose=False)
        rows.append(_row(s))
        r = rows[-1]
        print(f"  {team} {r['track']} {r['time'] if r['finished'] else 'DNF'}  cones={r['cones']} tech={r['tech']:.1f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", action="store_true", help="drive the analytic teacher instead of the network")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-perturb", action="store_true")
    ap.add_argument("--tracks", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_sim_config()
    cfg["output"]["plot"] = False
    tracks = [Path(p) for p in args.tracks] if args.tracks else list(PUBLIC)
    hold = ROOT / "tracks" / "generated" / "holdout"
    if hold.is_dir() and not args.tracks:
        tracks.extend(sorted(hold.glob("*.yaml")))
    extra = {"mode": "teacher"} if args.teacher else None
    team = "teacher" if args.teacher else "rl_longitudinal"
    print("== RL / teacher ==")
    rows = _drive(CTRL, tracks, team, cfg, controller_cfg=extra)
    if not args.no_baseline:
        print("== baseline ==")
        rows += _drive(BASE, tracks, "baseline", cfg)
    if not args.no_perturb:
        print("== perturbations on hackathon_test ==")
        ht = ROOT / "tracks" / "hackathon_test.yaml"
        for name, pert in PERTURB.items():
            if name == "nominal":
                continue
            rows += _drive(CTRL, [ht], f"{team}/{name}", cfg, perturb=pert, controller_cfg=extra)
    _print(rows)
    rl_rows = [r for r in rows if r["team"] == team and not r["perturb"]]
    base_rows = [r for r in rows if r["team"] == "baseline"]
    if rl_rows and base_rows:
        by = {r["track"]: r for r in base_rows}
        print("\nvs baseline (same tracks):")
        for r in rl_rows:
            b = by.get(r["track"])
            if not b:
                continue
            dt = (r["time"] - b["time"]) if (r["finished"] and b["finished"]) else float("nan")
            print(f"  {r['track']:<22} dt={dt:+.2f}s  dcones={r['cones']-b['cones']:+d}  "
                  f"dtech={r['tech']-b['tech']:+.1f}")


if __name__ == "__main__":
    main()

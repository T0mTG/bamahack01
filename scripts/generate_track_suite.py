"""Write a fixed-seed course suite under tracks/generated/.

From the repo root:

    python scripts/generate_track_suite.py
    python scripts/generate_track_suite.py --style technical --count 4 --seed-start 100 --out tracks/generated/train

Each course is produced by autocross.trackgen, checked with Track.validate, and drawn to a PNG
beside the YAML. Seeds are fixed, so running this again recreates the same layouts.

The default run writes:
  * the original 4 demo courses in tracks/generated/
  * a training pool in tracks/generated/train/
  * a holdout pool in tracks/generated/holdout/   (never used by training)
plus a manifest.json in each output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocross import trackgen
from autocross.plot import preview_track
from autocross.track import build, save

OUT = ROOT / "tracks" / "generated"

# kind, file name, seed, length, width, obstacles, laps (closed only)
SUITE = (
    ("open", "open_fast", 1, 650.0, 6.0, 0, 1),
    ("open", "open_obstacles", 2, 700.0, 5.5, 3, 1),
    ("open", "open_technical", 4, 520.0, 5.2, 2, 1),
    ("closed", "closed_loop", 3, 600.0, 6.0, 1, 2),
)

TRAIN = (
    ("fast", 20, 4),
    ("technical", 40, 4),
    ("braking", 60, 3),
    ("accel", 80, 3),
    ("obstacles", 100, 3),
    ("mixed", 120, 3),
    ("closed", 140, 2),
)
HOLDOUT = (
    ("fast", 200, 1),
    ("technical", 210, 1),
    ("braking", 220, 1),
    ("obstacles", 230, 1),
    ("closed", 240, 1),
)


def _write(spec: dict, directory: Path, name: str, extra: dict) -> dict:
    spec = dict(spec)
    spec["name"] = name
    track = build(spec)
    problems = track.validate()
    if problems:
        raise RuntimeError("; ".join(problems))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    save(spec, path)
    png = path.with_suffix(".png")
    preview_track(track, png)
    row = {"name": name, "file": str(path.relative_to(ROOT)), "length_m": round(track.length, 1),
           "closed": track.closed, "cones": len(track.cones), "obstacles": len(track.obstacle_cones),
           **extra}
    print(f"wrote {path}: {track.length:.0f} m, {len(track.cones)} cones, "
          f"{len(track.obstacle_cones)} obstacle cones, seed {extra.get('seed')}")
    return row


def _generate_one(style: str, seed: int, **kwargs) -> dict:
    last = None
    for attempt in range(8):
        use = seed + attempt * 97
        try:
            if style in ("open", "closed"):
                if style == "closed":
                    return trackgen.generate_closed(use, kwargs.get("length", 600.0), kwargs.get("width", 6.0),
                                                    kwargs.get("laps", 2), kwargs.get("obstacles", 1))
                return trackgen.generate_open(use, kwargs.get("length", 650.0), kwargs.get("width", 5.5),
                                              n_obstacles=kwargs.get("obstacles", 0))
            return trackgen.generate(use, style)
        except RuntimeError as exc:
            last = exc
    raise RuntimeError(f"could not generate {style} seed {seed}: {last}")


def write_demo() -> list[dict]:
    rows = []
    for kind, name, seed, length, width, obstacles, laps in SUITE:
        spec = _generate_one(kind, seed, length=length, width=width, obstacles=obstacles, laps=laps)
        rows.append(_write(spec, OUT, name, {"style": kind, "seed": seed}))
    return rows


def write_pool(directory: Path, plan: tuple, prefix: str) -> list[dict]:
    rows = []
    for style, seed0, count in plan:
        for i in range(count):
            seed = seed0 + i
            name = f"{prefix}_{style}_{seed}"
            spec = _generate_one(style, seed)
            rows.append(_write(spec, directory, name, {"style": style, "seed": seed, "split": prefix}))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=list(trackgen.STYLES), help="generate only this style (skips the default suite)")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=300)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    if args.style:
        out = args.out or OUT / "custom"
        rows = write_pool(out, ((args.style, args.seed_start, args.count),), "custom")
        (out / "manifest.json").write_text(json.dumps(rows, indent=2))
        return

    demo = write_demo()
    (OUT / "manifest.json").write_text(json.dumps(demo, indent=2))
    train = write_pool(OUT / "train", TRAIN, "train")
    (OUT / "train" / "manifest.json").write_text(json.dumps(train, indent=2))
    hold = write_pool(OUT / "holdout", HOLDOUT, "holdout")
    (OUT / "holdout" / "manifest.json").write_text(json.dumps(hold, indent=2))
    print(f"demo {len(demo)}, train {len(train)}, holdout {len(hold)}")


if __name__ == "__main__":
    main()

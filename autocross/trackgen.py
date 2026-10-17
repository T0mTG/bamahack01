"""Seeded procedural course generator. Output is a normal course YAML spec (segments for
open courses, explicit points for closed ones) so organisers can hand-edit the result."""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .track import build


STYLES = ("fast", "technical", "braking", "accel", "obstacles", "mixed", "closed")

_STYLE_P = {
    "fast":       [0.32, 0.32, 0.18, 0.04, 0.08, 0.06],
    "technical":  [0.12, 0.10, 0.28, 0.16, 0.18, 0.16],
    "braking":    [0.40, 0.10, 0.15, 0.25, 0.05, 0.05],
    "accel":      [0.38, 0.22, 0.22, 0.04, 0.08, 0.06],
    "obstacles":  [0.22, 0.18, 0.25, 0.10, 0.15, 0.10],
    "mixed":      [0.22, 0.20, 0.25, 0.10, 0.13, 0.10],
}


def _random_segment(rng: np.random.Generator, width: float, style: str = "mixed") -> dict:
    kind = rng.choice(["straight", "sweeper", "turn", "hairpin", "slalom", "chicane"],
                      p=_STYLE_P.get(style, _STYLE_P["mixed"]))
    side = float(rng.choice([-1.0, 1.0]))
    if kind == "straight":
        return {"type": "straight", "length": round(float(rng.uniform(12, 45)), 1)}
    if kind == "sweeper":
        return {"type": "arc", "radius": round(float(rng.uniform(25, 45)), 1),
                "angle": round(side * float(rng.uniform(30, 100)), 1)}
    if kind == "turn":
        return {"type": "arc", "radius": round(float(rng.uniform(12, 22)), 1),
                "angle": round(side * float(rng.uniform(60, 120)), 1)}
    if kind == "hairpin":
        return {"type": "arc", "radius": round(float(rng.uniform(9, 11.5)), 1),
                "angle": round(side * float(rng.uniform(150, 185)), 1), "width": round(width + 1.0, 1)}
    if kind == "slalom":
        return {"type": "slalom", "cycles": int(rng.integers(2, 4)), "wavelength": round(float(rng.uniform(22, 30)), 1),
                "swing": round(float(rng.uniform(2.0, 3.0)), 1), "first": "left" if side > 0 else "right"}
    return {"type": "slalom", "cycles": 1, "wavelength": round(float(rng.uniform(26, 34)), 1),
            "swing": round(float(rng.uniform(2.5, 3.5)), 1), "first": "left" if side > 0 else "right"}


def generate_open(seed: int, target_length: float = 700.0, width: float = 5.5,
                  max_extent: float = 320.0, n_obstacles: int = 0, attempts: int = 400,
                  style: str = "mixed") -> dict:
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        segs = [{"type": "straight", "length": 30.0}]
        length, failures = 30.0, 0
        while length < target_length - 60 and failures < 60:
            failures += 1
            s = _random_segment(rng, width, style)
            trial = segs + [s]
            spec = _spec(f"gen_open_{seed}", False, width, trial)
            try:
                t = build(spec)
            except ValueError:
                continue
            if t.validate() or _extent(t) > max_extent:
                continue
            segs, length = trial, t.length
        segs.append({"type": "straight", "length": 55.0})
        spec = _spec(f"gen_open_{seed}", False, width, segs)
        t = build(spec)
        if t.validate() or _extent(t) > max_extent or t.length < 0.7 * target_length:
            continue
        spec["obstacles"] = _obstacles(rng, t, n_obstacles)
        return spec
    raise RuntimeError("could not generate a valid course; try another seed or relax limits")


def generate_closed(seed: int, target_length: float = 600.0, width: float = 6.0,
                    laps: int = 2, n_obstacles: int = 0, attempts: int = 400) -> dict:
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    for _ in range(attempts):
        r = np.ones_like(th)
        for k in range(2, 8):
            r += rng.uniform(0.0, 0.45 / k ** 0.8) * np.cos(k * th + rng.uniform(0, 2 * math.pi))
        if r.min() < 0.25:
            continue
        pts = np.column_stack((r * np.cos(th), r * np.sin(th)))
        pts = g.smooth_closed(pts, 9)
        pts *= target_length / g.arclength(pts, True)[-1]
        if g.segments_intersect_any(pts[::6], closed=True):
            continue
        pts = g.resample(pts, 2.0, True)
        spec = {"name": f"gen_closed_{seed}", "closed": True, "laps": laps, "width": width, "cone_spacing": 5.0,
                "start_offset": 12.0, "centerline": [[round(float(x), 2), round(float(y), 2)] for x, y in pts]}
        t = build(spec)
        if t.validate():
            continue
        spec["obstacles"] = _obstacles(rng, t, n_obstacles)
        return spec
    raise RuntimeError("could not generate a valid closed course; try another seed")


def _template_braking(rng, width):
    side = float(rng.choice([-1.0, 1.0]))
    return [
        {"type": "straight", "length": round(float(rng.uniform(140, 200)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(9.5, 12.0)), 1),
         "angle": round(side * float(rng.uniform(155, 185)), 1), "width": round(width + 1.0, 1)},
        {"type": "straight", "length": round(float(rng.uniform(50, 90)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(16, 24)), 1),
         "angle": round(-side * float(rng.uniform(50, 90)), 1)},
        {"type": "straight", "length": 55.0},
    ]


def _template_accel(rng, width):
    side = float(rng.choice([-1.0, 1.0]))
    return [
        {"type": "straight", "length": round(float(rng.uniform(50, 80)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(16, 24)), 1),
         "angle": round(side * float(rng.uniform(70, 110)), 1)},
        {"type": "straight", "length": round(float(rng.uniform(70, 110)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(18, 28)), 1),
         "angle": round(-side * float(rng.uniform(60, 100)), 1)},
        {"type": "straight", "length": round(float(rng.uniform(60, 90)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(20, 30)), 1),
         "angle": round(side * float(rng.uniform(50, 80)), 1)},
        {"type": "straight", "length": 55.0},
    ]


def _template_technical(rng, width):
    side = float(rng.choice([-1.0, 1.0]))
    kink_w = round(max(width - 0.4, 4.8), 1)
    return [
        {"type": "straight", "length": 30.0},
        {"type": "straight", "length": round(float(rng.uniform(50, 80)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(55, 80)), 1),
         "angle": round(-side * float(rng.uniform(12, 20)), 1), "width": kink_w},
        {"type": "arc", "radius": round(float(rng.uniform(55, 80)), 1),
         "angle": round(side * float(rng.uniform(12, 20)), 1), "width": kink_w},
        {"type": "straight", "length": round(float(rng.uniform(35, 55)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(9.5, 11.5)), 1),
         "angle": round(-side * float(rng.uniform(110, 140)), 1), "width": round(width + 1.0, 1)},
        {"type": "arc", "radius": round(float(rng.uniform(22, 30)), 1),
         "angle": round(-side * float(rng.uniform(35, 55)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(28, 38)), 1),
         "angle": round(side * float(rng.uniform(30, 50)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(16, 22)), 1),
         "angle": round(side * float(rng.uniform(40, 60)), 1)},
        {"type": "arc", "radius": round(float(rng.uniform(10, 13)), 1),
         "angle": round(side * float(rng.uniform(55, 80)), 1), "width": round(width + 1.0, 1)},
        {"type": "straight", "length": 20.0},
        {"type": "arc", "radius": 13.0, "angle": round(-side * 60, 1)},
        {"type": "arc", "radius": 13.0, "angle": round(side * 120, 1)},
        {"type": "arc", "radius": 13.0, "angle": round(-side * 60, 1)},
        {"type": "straight", "length": 55.0},
    ]


def _from_segments(name, width, segs, rng, n_obstacles, max_extent=360.0):
    spec = _spec(name, False, width, segs)
    t = build(spec)
    if t.validate() or _extent(t) > max_extent:
        raise ValueError("invalid template course")
    spec["obstacles"] = _obstacles(rng, t, n_obstacles)
    return spec


def generate(seed: int, style: str = "mixed", target_length: float = 650.0, width: float = 5.5,
             n_obstacles: int | None = None, laps: int = 2, attempts: int = 80) -> dict:
    """Build a course of the given style. ``style`` is one of STYLES."""
    style = (style or "mixed").lower()
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    rng = np.random.default_rng(seed)
    if style == "closed":
        return generate_closed(seed, target_length, width, laps, 1 if n_obstacles is None else n_obstacles)
    if n_obstacles is None:
        n_obstacles = {"fast": 0, "technical": 3, "braking": 0, "accel": 0, "obstacles": 4, "mixed": 2}[style]
    templates = {"braking": _template_braking, "accel": _template_accel, "technical": _template_technical}
    last = None
    if style in templates:
        for k in range(attempts):
            try:
                return _from_segments(f"gen_{style}_{seed}", width, templates[style](np.random.default_rng(seed + 17 * k), width),
                                      np.random.default_rng(seed + 31 * k), n_obstacles)
            except (ValueError, RuntimeError) as exc:
                last = exc
        raise RuntimeError(f"could not generate a {style} course: {last}")
    return generate_open(seed, target_length, width, n_obstacles=n_obstacles, style=style)


def _spec(name, closed, width, segs):
    return {"name": name, "closed": closed, "width": width, "cone_spacing": 5.0, "start_offset": 8.0,
            "finish_runoff": 40.0, "origin": [0.0, 0.0, 0.0], "segments": segs}


def _extent(t) -> float:
    x0, y0, x1, y1 = t.bounds()
    return max(x1 - x0, y1 - y0)


def _obstacles(rng, t, n) -> list:
    if n <= 0:
        return []
    lo, hi = t.start_progress + 60.0, t.finish_progress - 40.0
    if t.closed:
        lo, hi = t.start_offset + 60.0, t.start_offset + t.length - 40.0
    out = []
    for s in np.sort(rng.uniform(lo, hi, n)):
        wl, wr = t.width_at(t.s_of_progress(float(s)))
        side = float(rng.choice([-1.0, 1.0]))
        w = float(rng.uniform(1.2, 2.0))
        off = side * ((wl if side > 0 else wr) - w / 2 - 0.3)  # outer cone just inside the boundary line
        out.append({"s": round(float(t.s_of_progress(float(s))), 1), "offset": round(off, 2), "width": round(w, 2)})
    return out

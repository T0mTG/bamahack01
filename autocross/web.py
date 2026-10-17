"""Browser for the bicycle model: pick a controller and a course, then replay the lap.

    python -m autocross web
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import ROOT, load_sim_config, load_yaml
from .runner import LOG_COLS
from .track import load as load_track

FRONTEND = ROOT / "frontend"
CACHE = ROOT / "runs" / "web_cache"
_LOCK = threading.Lock()

CONTROLLERS = (
    {
        "id": "baseline",
        "name": "Baseline",
        "blurb": "Pure pursuit on the centerline. Ignores obstacles.",
        "file": "controllers/baseline_pure_pursuit.py",
    },
    {
        "id": "rl_longitudinal",
        "name": "Longitudinal RL",
        "blurb": "Obstacle-aware path. A network sets throttle and brake.",
        "file": "controllers/rl_longitudinal/controller.py",
    },
    {
        "id": "rl_longitudinal_67",
        "name": "Longitudinal RL 67",
        "blurb": "Same planner, earlier network weights.",
        "file": "controllers/rl_longitudinal_67/controller.py",
    },
    {
        "id": "rl_drift",
        "name": "Drift RL",
        "blurb": "Steering residual and a later brake, so the car can slide.",
        "file": "controllers/rl_drift/controller.py",
    },
)

TRACK_ORDER = (
    "hackathon_test",
    "practice_basic",
    "practice_technical",
    "practice_obstacles",
    "practice_loop",
    "brake_hairpin",
    "accel_straights",
    "generated/open_fast",
    "generated/open_technical",
    "generated/open_obstacles",
    "generated/closed_loop",
)

TRACK_TITLES = {
    "hackathon_test": "Hackathon test",
    "practice_basic": "Practice basic",
    "practice_technical": "Practice technical",
    "practice_obstacles": "Practice obstacles",
    "practice_loop": "Practice loop",
    "brake_hairpin": "Brake hairpin",
    "accel_straights": "Accel straights",
    "generated/open_fast": "Open fast",
    "generated/open_technical": "Open technical",
    "generated/open_obstacles": "Open obstacles",
    "generated/closed_loop": "Closed loop",
}

TRACK_BLURBS = {
    "hackathon_test": "Public development course.",
    "practice_basic": "Open layout with a hairpin and a slalom.",
    "practice_technical": "Kink, hairpin, decreasing radius, obstacles.",
    "practice_obstacles": "Unmapped cones the car has to avoid.",
    "practice_loop": "Closed course, two laps.",
    "brake_hairpin": "Long straight into a tight hairpin.",
    "accel_straights": "Straights broken by medium corners.",
    "generated/open_fast": "Generated open course, fast style.",
    "generated/open_technical": "Generated open course, technical style.",
    "generated/open_obstacles": "Generated open course with obstacles.",
    "generated/closed_loop": "Generated closed loop.",
}


class UnknownChoice(LookupError):
    pass


def _controllers():
    out = []
    for item in CONTROLLERS:
        path = (ROOT / item["file"]).resolve()
        if path.is_file():
            out.append({**item, "file": str(path)})
    return out


def _tracks():
    root = ROOT / "tracks"
    found = {}
    paths = list(root.glob("*.yaml"))
    gen = root / "generated"
    if gen.is_dir():
        paths += list(gen.glob("*.yaml"))
    for path in paths:
        rel = path.relative_to(root).as_posix()
        tid = rel[: -len(".yaml")] if rel.endswith(".yaml") else rel
        spec = load_yaml(path)
        found[tid] = {
            "id": tid,
            "name": TRACK_TITLES.get(tid, str(spec.get("name") or path.stem).replace("_", " ")),
            "blurb": TRACK_BLURBS.get(tid, "Generated course."),
            "group": "Generated" if "/" in tid else "Public",
            "closed": bool(spec.get("closed", False)),
            "laps": int(spec.get("laps") or 1),
            "file": str(path.resolve()),
        }
    order = {name: i for i, name in enumerate(TRACK_ORDER)}
    return sorted(found.values(), key=lambda t: (order.get(t["id"], 10_000), t["id"]))


def catalog() -> dict:
    controllers = [{k: c[k] for k in ("id", "name", "blurb")} for c in _controllers()]
    tracks = [{k: t[k] for k in ("id", "name", "blurb", "group", "closed", "laps")} for t in _tracks()]
    return {"controllers": controllers, "tracks": tracks}


def _controller(controller_id: str) -> dict:
    for item in _controllers():
        if item["id"] == controller_id:
            return item
    raise UnknownChoice(f"unknown controller '{controller_id}'")


def _track(track_id: str) -> dict:
    for item in _tracks():
        if item["id"] == track_id:
            return item
    raise UnknownChoice(f"unknown map '{track_id}'")


def _pts(arr, nd: int = 2) -> list:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return []
    if len(a) > 2500:
        a = a[:: max(1, len(a) // 1600)]
    return [[round(float(p[0]), nd), round(float(p[1]), nd)] for p in a]


def course_document(track_id: str) -> dict:
    """Geometry the page draws before a controller has driven the course."""
    meta = _track(track_id)
    track = load_track(meta["file"])
    cfg = load_sim_config()
    bike = cfg["vehicle"]["bicycle"]
    pos, col, mapped = track.all_cones()
    cones = [
        [round(float(p[0]), 2), round(float(p[1]), 2), int(c), int(bool(m))]
        for p, c, m in zip(pos, col, mapped)
    ]
    sx, sy, syaw = track.spawn_pose()
    start = track.cross_line(track.s_of_progress(track.start_progress))
    finish = track.cross_line(track.s_of_progress(track.finish_progress))
    bounds = track.bounds(8.0)
    return {
        "id": meta["id"],
        "name": meta["name"],
        "closed": bool(track.closed),
        "laps": int(track.laps),
        "length": round(float(track.length), 2),
        "start": round(float(track.start_progress), 2),
        "finish": round(float(track.finish_progress), 2),
        "bounds": [round(float(v), 2) for v in bounds],
        "center": _pts(track.center),
        "left": _pts(track.left),
        "right": _pts(track.right),
        "cones": cones,
        "startLine": _pts(start),
        "finishLine": _pts(finish),
        "spawn": {"x": round(sx, 3), "y": round(sy, 3), "yaw": round(float(syaw), 4)},
        "vehicle": {
            "length": float(bike["length"]),
            "width": float(bike["width"]),
            "wheelbase": float(bike["wheelbase"]),
        },
    }


def _drive_from_csv(path: Path) -> dict:
    want = ("t", "x", "y", "yaw", "speed", "steer_angle", "progress")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = [header.index(name) for name in want]
        rows = [[float(row[i]) for i in idx] for row in reader if row and row[0] != ""]
    if not rows:
        return {"dt": 0.05, "x": [], "y": [], "yaw": [], "v": [], "steer": [], "s": []}
    arr = np.asarray(rows, dtype=float)
    dt = float(arr[1, 0] - arr[0, 0]) if len(arr) > 1 else 0.025
    stride = max(1, int(round(0.05 / dt))) if dt > 0 else 1
    sl = arr[::stride]

    def col(i: int, nd: int) -> list:
        return [round(float(v), nd) for v in sl[:, i]]

    return {
        "dt": round(dt * stride, 4),
        "x": col(1, 3),
        "y": col(2, 3),
        "yaw": col(3, 4),
        "v": col(4, 3),
        "steer": col(5, 4),
        "s": col(6, 2),
    }


def _events(path: Path) -> list:
    raw = json.loads(path.read_text(encoding="utf-8"))
    keep = {"cone_hit", "off_course", "missed_gate"}
    return [
        {
            "t": round(float(e["t"]), 3),
            "kind": e["kind"],
            "x": round(float(e["x"]), 2),
            "y": round(float(e["y"]), 2),
        }
        for e in raw
        if e.get("kind") in keep
    ]


def _score(summary: dict) -> dict:
    comp = summary["metrics"]["completion"]
    pen = summary["metrics"]["penalties"]
    sc = summary["score"]
    time_s = comp["time_s"]
    return {
        "finished": bool(comp["finished"]),
        "time_s": None if time_s is None else round(float(time_s), 2),
        "dnf": comp.get("dnf_reason"),
        "progress": round(float(comp["progress_fraction"]), 4),
        "cones": int(pen["cone_hit"]),
        "off_course": int(pen["off_course"]),
        "missed_gates": int(pen["missed_gate"]),
        "technical": float(sc["technical_points"]),
        "technical_max": float(sc["technical_max"]),
        "adjusted_time_s": sc.get("adjusted_time_s"),
        "reference_time_s": None if comp.get("reference_time_s") is None else round(float(comp["reference_time_s"]), 2),
        "categories": {k: float(v) for k, v in sc.get("category_points", {}).items()},
    }


def _cache_path(controller_id: str, track_id: str) -> Path:
    safe = track_id.replace("/", "__")
    return CACHE / f"{controller_id}__{safe}.json"


def run_replay(controller_id: str, track_id: str, *, force: bool = False) -> dict:
    ctrl = _controller(controller_id)
    meta = _track(track_id)
    cache = _cache_path(controller_id, track_id)
    with _LOCK:
        if cache.is_file() and not force:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            payload["cached"] = True
            print(f"[web] {ctrl['name']} on {meta['name']} (saved lap)", flush=True)
            return payload
        print(f"[web] simulating {ctrl['name']} on {meta['name']}", flush=True)
        from .runner import run

        cfg = load_sim_config()
        cfg["output"]["plot"] = False
        cfg["carla"]["render"] = False
        tmp = Path(tempfile.mkdtemp(prefix="milsim-"))
        try:
            summary = run(
                cfg,
                ctrl["file"],
                meta["file"],
                team=ctrl["id"],
                backend="bicycle",
                out_dir=tmp / "run",
                verbose=False,
            )
            out = Path(summary["out_dir"])
            payload = {
                "cached": False,
                "controller": {"id": ctrl["id"], "name": ctrl["name"]},
                "track": {"id": meta["id"], "name": meta["name"]},
                "course": course_document(track_id),
                "drive": _drive_from_csv(out / "telemetry.csv"),
                "events": _events(out / "events.json"),
                "score": _score(summary),
            }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
        sc = payload["score"]
        status = f"finished {sc['time_s']:.2f} s" if sc["finished"] and sc["time_s"] is not None else f"DNF {sc['dnf']}"
        print(f"[web] {ctrl['name']} on {meta['name']} -> {status}", flush=True)
        return payload


_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        message = fmt % args
        if "/api/" in message or message.startswith("POST"):
            super().log_message(fmt, *args)

    def _json(self, obj, status: int = 200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 8192:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/catalog":
                self._json(catalog())
                return
            if parsed.path == "/api/course":
                track_id = (parse_qs(parsed.query).get("track") or [""])[0]
                if not track_id:
                    self._json({"error": "missing track"}, 400)
                    return
                self._json(course_document(track_id))
                return
            self._static(parsed.path)
        except UnknownChoice as e:
            self._json({"error": str(e)}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = self._read_json()
            controller_id = str(body.get("controller") or "")
            track_id = str(body.get("track") or "")
            if not controller_id or not track_id:
                self._json({"error": "controller and track are required"}, 400)
                return
            self._json(run_replay(controller_id, track_id, force=bool(body.get("force"))))
        except UnknownChoice as e:
            self._json({"error": str(e)}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)

    def _static(self, url_path: str):
        rel = url_path.lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        root = FRONTEND.resolve()
        target = (FRONTEND / rel).resolve()
        if target != root and root not in target.parents:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"MIL-SIM  {url}", flush=True)
    print("Choose a controller and a map, then Drive. The bicycle model runs on this machine. Ctrl+C stops the server.", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()

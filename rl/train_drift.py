"""Train a pedal + steer residual on top of the rl_longitudinal_67 policy.

    python -m rl.train_drift
    python -m rl.train_drift --minutes 10 --device cuda

The pedal head is copied from ``controllers/rl_longitudinal_67/policy.npz``. The steer head
starts at zero, so the first checkpoint is that pedal policy plus pure pursuit, on the slightly
looser drift shield. PPO then has about 10 minutes to find out whether a controlled slide is
faster. The saved file is the best holdout checkpoint, as a numpy ``.npz``.
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRIFT_DIR = ROOT / "controllers" / "rl_drift"
DEFAULT_OUT = DRIFT_DIR / "policy.npz"
SOURCE_67 = ROOT / "controllers" / "rl_longitudinal_67" / "policy.npz"
ACT_DIM = 2

# Filled by _import_drift(). Kept off import time so a sandbox child, which re-imports this
# file, does not put the drift modules in front of rl_longitudinal_67.
FEATURE_DIM = 35
MLPPolicy = combine_u = NOMINAL = DriftEnv = run_episode = None
_LOADED = False


def _import_drift():
    global _LOADED, FEATURE_DIM, MLPPolicy, combine_u, NOMINAL, DriftEnv, run_episode
    if _LOADED:
        return
    if str(DRIFT_DIR) in sys.path:
        sys.path.remove(str(DRIFT_DIR))
    sys.path.insert(0, str(DRIFT_DIR))
    from lr_features import FEATURE_DIM as _FEATURE_DIM
    from lr_net import MLPPolicy as _MLPPolicy
    from lr_planner import combine_u as _combine_u
    from rl.env_drift import NOMINAL as _NOMINAL
    from rl.env_drift import DriftEnv as _DriftEnv
    from rl.env_drift import run_episode as _run_episode
    FEATURE_DIM = _FEATURE_DIM
    MLPPolicy = _MLPPolicy
    combine_u = _combine_u
    NOMINAL = _NOMINAL
    DriftEnv = _DriftEnv
    run_episode = _run_episode
    _LOADED = True


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def default_tracks() -> list[Path]:
    tracks = [
        ROOT / "tracks" / "brake_hairpin.yaml",
        ROOT / "tracks" / "practice_technical.yaml",
        ROOT / "tracks" / "hackathon_test.yaml",
        ROOT / "tracks" / "accel_straights.yaml",
        ROOT / "tracks" / "generated" / "train" / "train_technical_40.yaml",
        ROOT / "tracks" / "generated" / "train" / "train_technical_41.yaml",
    ]
    found = [p for p in tracks if p.is_file()]
    if not found:
        raise SystemExit("no course YAML found to train on")
    return found


def holdout_tracks() -> list[Path]:
    names = ["holdout_technical_210.yaml", "holdout_braking_220.yaml", "holdout_fast_200.yaml"]
    return [p for p in (ROOT / "tracks" / "generated" / "holdout" / n for n in names) if p.is_file()]


def expand_from_67(path: Path, obs_dim: int | None = None) -> MLPPolicy:
    _import_drift()
    if obs_dim is None:
        obs_dim = FEATURE_DIM
    """Copy the 67 pedal network and add a zero steer output plus zero columns for new inputs."""
    old = MLPPolicy.load(path)
    old_dim = int(old.weights[0].shape[0])
    if obs_dim < old_dim:
        raise SystemExit(f"drift features ({obs_dim}) are shorter than the 67 policy ({old_dim})")
    hidden = int(old.weights[0].shape[1])
    weights, biases = [], []
    w0 = np.zeros((obs_dim, hidden))
    w0[:old_dim] = old.weights[0]
    weights.append(w0)
    biases.append(np.array(old.biases[0], copy=True))
    for w, b in zip(old.weights[1:-1], old.biases[1:-1]):
        weights.append(np.array(w, copy=True))
        biases.append(np.array(b, copy=True))
    w_last = np.zeros((old.weights[-1].shape[0], ACT_DIM))
    w_last[:, :1] = np.asarray(old.weights[-1]).reshape(old.weights[-1].shape[0], -1)[:, :1]
    b_last = np.zeros(ACT_DIM)
    b_last[0] = float(np.reshape(old.biases[-1], -1)[0])
    weights.append(w_last)
    biases.append(b_last)
    return MLPPolicy(weights, biases, np.zeros(obs_dim), np.ones(obs_dim))


def _normalize(x, mean, std):
    return np.clip((x - mean) / std, -6.0, 6.0)


def _build_models(torch, obs_dim, hidden, device):
    class MLP(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, out),
            )

        def forward(self, x):
            return self.net(x)

    actor, critic = MLP(ACT_DIM).to(device), MLP(1).to(device)
    log_std = torch.nn.Parameter(torch.tensor([-1.4, -2.0], device=device))
    return actor, critic, log_std


def _load_actor(torch, actor, policy: MLPPolicy):
    linears = [m for m in actor.modules() if isinstance(m, torch.nn.Linear)]
    if len(linears) != len(policy.weights):
        raise RuntimeError("actor depth does not match the warm-start policy")
    with torch.no_grad():
        for lin, w, b in zip(linears, policy.weights, policy.biases):
            lin.weight.copy_(torch.tensor(w.T, dtype=lin.weight.dtype, device=lin.weight.device))
            lin.bias.copy_(torch.tensor(b, dtype=lin.bias.dtype, device=lin.bias.device))


def _actor_arrays(actor):
    torch = _torch()
    weights, biases = [], []
    for mod in actor.modules():
        if isinstance(mod, torch.nn.Linear):
            weights.append(mod.weight.detach().cpu().numpy().T.copy())
            biases.append(mod.bias.detach().cpu().numpy().copy())
    mean, std = np.zeros(weights[0].shape[0]), np.ones(weights[0].shape[0])
    return weights, biases, mean, std


def _policy_from_actor(actor) -> MLPPolicy:
    return MLPPolicy(*_actor_arrays(actor))


def _apply_action(env, residual: np.ndarray) -> tuple[float, float]:
    _import_drift()
    residual = np.asarray(residual, dtype=float).reshape(-1)
    pedal = float(residual[0]) if residual.size else 0.0
    steer_r = float(residual[1]) if residual.size > 1 else 0.0
    return combine_u(env.teacher_action(), pedal), steer_r


def _gae(rew, val, done, last_v, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rew)
    running = 0.0
    for t in reversed(range(len(rew))):
        nxt = last_v if t == len(rew) - 1 else val[t + 1]
        alive = 1.0 - done[t]
        delta = rew[t] + gamma * nxt * alive - val[t]
        running = delta + gamma * lam * alive * running
        adv[t] = running
    return adv, adv + val


def _squashed_logp(torch, pre, loc, log_std):
    std = torch.exp(log_std)
    return (-0.5 * ((pre - loc) / std) ** 2 - log_std
            - torch.log(1.0 - torch.tanh(pre) ** 2 + 1e-6)).sum(-1)


def validate(policy, tracks, seed=0) -> dict:
    _import_drift()
    env = DriftEnv(randomize=False, seed=seed + 9)

    def act(feat, e):
        return _apply_action(e, policy.act(feat))

    rows = [run_episode(env, act, str(p), seed=seed, rand=NOMINAL) for p in tracks]
    fin = [r for r in rows if r["finished"]]
    times = [r["time"] / max(r["reference"], 1e-3) for r in fin]
    cones = sum(r["cones"] for r in rows)
    score = (100.0 * (len(fin) / max(len(rows), 1))
             - 8.0 * sum(1 for r in rows if r["dnf"])
             - 2.0 * cones
             - 20.0 * (float(np.mean(times)) - 1.0 if times else 1.0))
    return {"score": score, "finish": len(fin), "n": len(rows), "cones": cones,
            "ratio": float(np.mean(times)) if times else float("nan"),
            "slip": float(np.mean([r["slip_frac"] for r in rows])) if rows else 0.0,
            "drift": float(np.mean([r["drift_frac"] for r in rows])) if rows else 0.0,
            "rows": rows}


def _worker_loop(conn, seed, track_paths, max_time_s):
    _import_drift()
    env = DriftEnv(randomize=True, max_time_s=max_time_s, seed=seed)
    tracks = [str(p) for p in track_paths]
    rng = np.random.default_rng(seed)
    policy = None
    env.reset(track=tracks[0], seed=seed)
    while True:
        msg = conn.recv()
        cmd = msg["cmd"]
        if cmd == "stop":
            break
        if cmd == "set":
            policy = MLPPolicy(msg["weights"], msg["biases"], msg["mean"], msg["std"])
            continue
        n = int(msg["n"])
        std_action = np.asarray(msg["std_action"], dtype=float)
        obs, act, rew, done = [], [], [], []
        ep_info = []
        ep_slip = ep_drift = 0.0
        ep_steps = 0
        for _ in range(n):
            feat = env._feat
            if policy is None:
                mean = np.zeros(ACT_DIM)
            else:
                mean = np.clip(np.asarray(policy.act(feat), dtype=float).reshape(-1), -0.999, 0.999)
            pre = np.arctanh(mean) + std_action * rng.normal(size=ACT_DIM)
            residual = np.tanh(pre)
            u, steer_r = _apply_action(env, residual)
            _, r, d, info = env.step(u, steer_r)
            obs.append(np.array(feat, copy=True))
            act.append(pre)
            rew.append(r)
            done.append(float(d))
            ep_slip += info["slip"]
            ep_drift += info["drift"]
            ep_steps += 1
            if d:
                finished = dict(info)
                finished["slip"] = ep_slip / ep_steps
                finished["drift"] = ep_drift / ep_steps
                ep_info.append(finished)
                ep_slip = ep_drift = 0.0
                ep_steps = 0
                env.reset(track=tracks[int(rng.integers(0, len(tracks)))])
        conn.send({"obs": np.asarray(obs), "act": np.asarray(act), "rew": np.asarray(rew),
                   "done": np.asarray(done), "ep_info": ep_info})


def _spawn_workers(n, tracks, max_time_s, seed):
    ctx = mp.get_context("spawn")
    workers = []
    for i in range(n):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker_loop, args=(b, seed + 1000 * (i + 1), tracks, max_time_s), daemon=True)
        p.start()
        workers.append((p, a))
    return workers


def _broadcast(workers, payload):
    for _, conn in workers:
        conn.send(payload)


def _collect(workers, n_each, std_action):
    for _, conn in workers:
        conn.send({"cmd": "rollout", "n": n_each, "std_action": std_action})
    batches = [conn.recv() for _, conn in workers]
    return {
        "obs": np.concatenate([b["obs"] for b in batches], 0),
        "act": np.concatenate([b["act"] for b in batches], 0),
        "rew": np.concatenate([b["rew"] for b in batches], 0),
        "done": np.concatenate([b["done"] for b in batches], 0),
        "ep_info": sum((b["ep_info"] for b in batches), []),
    }


def _stop_workers(procs):
    if not procs:
        return
    _broadcast(procs, {"cmd": "stop"})
    for p, _ in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()


def train_torch(tracks, holdout, out, minutes, workers, device, hidden, seed, rollout, max_time_s, lr):
    _import_drift()
    torch = _torch()
    device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"torch {torch.__version__} device={device} workers={workers}")
    actor, critic, log_std = _build_models(torch, FEATURE_DIM, hidden, device)
    if not SOURCE_67.is_file():
        raise SystemExit(f"missing warm-start weights: {SOURCE_67}")
    warm = expand_from_67(SOURCE_67, FEATURE_DIM)
    _load_actor(torch, actor, warm)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=lr)
    mean, std = np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)
    warm.save(out)
    print(f"wrote warm-start {out}")
    val0 = validate(warm, holdout, seed=seed)
    print(f"  warm-start validate score {val0['score']:.1f}  finish {val0['finish']}/{val0['n']}  "
          f"cones {val0['cones']}  time/ref {val0['ratio']:.3f}  slip {val0['slip']:.3f}  drift {val0['drift']:.3f}")
    best_score = val0["score"]
    best_state = (copy.deepcopy(warm.weights), copy.deepcopy(warm.biases), mean.copy(), std.copy())

    procs = _spawn_workers(max(workers, 1), tracks, max_time_s, seed) if workers > 1 else None
    t0 = time.perf_counter()
    deadline = t0 + minutes * 60.0
    steps_done = 0
    update = 0
    rng = np.random.default_rng(seed)
    env = DriftEnv(randomize=True, max_time_s=max_time_s, seed=seed)

    try:
        while time.perf_counter() < deadline:
            ws, bs, mu, sd = _actor_arrays(actor)
            std_action = torch.exp(log_std).detach().cpu().numpy()
            if procs:
                _broadcast(procs, {"cmd": "set", "weights": ws, "biases": bs, "mean": mu, "std": sd})
                batch = _collect(procs, max(rollout // max(workers, 1), 1), std_action.tolist())
                obs_np, act_np, rew_np, done_np = batch["obs"], batch["act"], batch["rew"], batch["done"]
                infos = batch["ep_info"]
            else:
                obs_np, act_np, rew_np, done_np, infos = [], [], [], [], []
                ep_slip = ep_drift = 0.0
                ep_steps = 0
                env.reset(track=str(tracks[int(rng.integers(0, len(tracks)))]))
                for _ in range(rollout):
                    feat = env._feat.copy()
                    loc = actor(torch.tensor(_normalize(feat, mu, sd), dtype=torch.float32, device=device)).detach().cpu().numpy()
                    pre = loc + std_action * rng.normal(size=ACT_DIM)
                    residual = np.tanh(pre)
                    u, steer_r = _apply_action(env, residual)
                    _, r, d, info = env.step(u, steer_r)
                    obs_np.append(feat)
                    act_np.append(pre)
                    rew_np.append(r)
                    done_np.append(float(d))
                    ep_slip += info["slip"]
                    ep_drift += info["drift"]
                    ep_steps += 1
                    if d:
                        finished = dict(info)
                        finished["slip"] = ep_slip / ep_steps
                        finished["drift"] = ep_drift / ep_steps
                        infos.append(finished)
                        ep_slip = ep_drift = 0.0
                        ep_steps = 0
                        env.reset(track=str(tracks[int(rng.integers(0, len(tracks)))]))
                obs_np, act_np, rew_np, done_np = map(np.asarray, (obs_np, act_np, rew_np, done_np))

            x = torch.tensor(_normalize(obs_np, mean, std), dtype=torch.float32, device=device)
            with torch.no_grad():
                values = critic(x).squeeze(-1).cpu().numpy()
                loc0 = actor(x)
            last_v = 0.0 if done_np[-1] else float(values[-1])
            adv, ret = _gae(rew_np, values, done_np, last_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            u_pre = torch.tensor(act_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                old_logp = _squashed_logp(torch, u_pre, loc0, log_std)
            adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
            ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

            last_loss = 0.0
            n = len(adv)
            for _ in range(6):
                order = torch.randperm(n, device=device)
                for s in range(0, n, 256):
                    b = order[s:s + 256]
                    if len(b) < 8:
                        continue
                    loc_b = actor(x[b])
                    logp = _squashed_logp(torch, u_pre[b], loc_b, log_std)
                    ratio = torch.exp(torch.clamp(logp - old_logp[b], -20, 20))
                    pol = -torch.mean(torch.minimum(ratio * adv_t[b], torch.clamp(ratio, 0.8, 1.2) * adv_t[b]))
                    vf = 0.5 * torch.mean((critic(x[b]).squeeze(-1) - ret_t[b]) ** 2)
                    ent = 0.5 * ACT_DIM * (1.0 + np.log(2.0 * np.pi)) + torch.sum(log_std)
                    loss = pol + 0.5 * vf - 0.01 * ent
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [log_std], 1.0)
                    opt.step()
                    with torch.no_grad():
                        log_std.clamp_(-2.5, 0.2)
                    last_loss = float(loss.item())

            steps_done += n
            update += 1
            finishes = sum(1 for i in infos if i.get("finished"))
            dnfs = sum(1 for i in infos if i.get("dnf"))
            slip = float(np.mean([i.get("slip", 0.0) for i in infos])) if infos else 0.0
            drift = float(np.mean([i.get("drift", 0.0) for i in infos])) if infos else 0.0
            elapsed = time.perf_counter() - t0
            print(f"  steps {steps_done:7d}  loss {last_loss:8.3f}  episodes {len(infos):3d}  "
                  f"finish {finishes}  dnf {dnfs}  slip {slip:.3f}  drift {drift:.3f}  {elapsed:.0f}s")

            if update % 4 == 0 or time.perf_counter() + 30 > deadline:
                pol = _policy_from_actor(actor)
                val = validate(pol, holdout, seed=seed)
                print(f"  validate score {val['score']:.1f}  finish {val['finish']}/{val['n']}  "
                      f"cones {val['cones']}  time/ref {val['ratio']:.3f}  slip {val['slip']:.3f}  "
                      f"drift {val['drift']:.3f}")
                if val["score"] >= best_score:
                    best_score = val["score"]
                    ws, bs, mu, sd = _actor_arrays(actor)
                    best_state = (copy.deepcopy(ws), copy.deepcopy(bs), mu.copy(), sd.copy())
                    MLPPolicy(*best_state).save(out)
                    print(f"  wrote {out} (best so far)")
    finally:
        _stop_workers(procs)

    if best_state:
        MLPPolicy(*best_state).save(out)
    print(f"wrote {out}  best_validate={best_score:.1f}  steps={steps_done}")


def compare(tracks=None) -> None:
    """Official bicycle runs of the drift controller against rl_longitudinal_67."""
    from autocross.config import load_sim_config
    from autocross.runner import run

    cfg = load_sim_config()
    tracks = tracks or [
        ROOT / "tracks" / "brake_hairpin.yaml",
        ROOT / "tracks" / "practice_technical.yaml",
        ROOT / "tracks" / "hackathon_test.yaml",
    ]
    controllers = [
        ("rl_longitudinal_67", ROOT / "controllers" / "rl_longitudinal_67" / "controller.py"),
        ("rl_drift", DRIFT_DIR / "controller.py"),
    ]
    print(f"{'controller':<22} {'track':<22} {'fin':>3} {'time':>7} {'ref':>6} "
          f"{'cone':>4} {'off':>3} {'tech':>6} {'slip%':>6} {'slipmax':>7}")
    for team, controller in controllers:
        for tp in tracks:
            summary = run(cfg, str(controller), str(tp), team=team, backend="bicycle", verbose=False)
            m, sc = summary["metrics"], summary["score"]
            c, p, dyn = m["completion"], m["penalties"], m["dynamics"]
            t = f"{c['time_s']:.1f}" if c["finished"] else "DNF"
            print(f"{team:<22} {summary['track']['name']:<22} {int(c['finished']):>3} {t:>7} "
                  f"{c['reference_time_s']:6.1f} {p['cone_hit']:4d} {p['off_course']:3d} "
                  f"{sc['technical_points']:6.1f} {100 * dyn['rear_slip_over_frac']:6.1f} "
                  f"{dyn['rear_slip_max_deg']:7.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", nargs="*", help="course YAML files")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--max-time", type=float, default=80.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--compare-only", action="store_true", help="skip training and run the bicycle comparison")
    ap.add_argument("--no-compare", action="store_true", help="skip the post-training comparison")
    args = ap.parse_args(argv)
    if args.compare_only:
        compare()
        return
    torch = _torch()
    if torch is None:
        raise SystemExit("torch is not installed. Use the CUDA venv, for example "
                         "E:\\ecocar_rl\\venv\\Scripts\\python.exe -m rl.train_drift")
    tracks = [Path(p) for p in args.tracks] if args.tracks else default_tracks()
    missing = [str(p) for p in tracks if not Path(p).is_file()]
    if missing:
        raise SystemExit("missing courses:\n  " + "\n  ".join(missing))
    hold = holdout_tracks()
    if not hold:
        raise SystemExit("no holdout courses found")
    print("training on")
    for p in tracks:
        print(f"  {p}")
    train_torch(tracks, hold, Path(args.out), args.minutes, args.workers, args.device,
                args.hidden, args.seed, args.rollout, args.max_time, args.lr)
    if not args.no_compare:
        compare()


if __name__ == "__main__":
    mp.freeze_support()
    main()

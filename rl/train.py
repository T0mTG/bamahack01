"""Train the longitudinal throttle/brake policy.

    python -m rl.train
    python -m rl.train --minutes 20 --device cuda --out controllers/rl_longitudinal/policy.npz

Rollouts run on CPU (the bicycle model). If torch+CUDA is available the PPO update uses the GPU;
otherwise torch-CPU or a numpy imitation fallback is used. The exported file is always a numpy
``.npz`` that the submission controller can load with no torch installed.

The policy is warm-started by imitating the analytic teacher, then PPO fine-tunes it. The best
checkpoint is chosen by a holdout validation score (finish rate, time vs reference, cone hits).
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import rl  # noqa: F401
from lr_features import FEATURE_DIM
from lr_net import MLPPolicy
from lr_planner import PlannerParams, combine_u
from rl.env import NOMINAL, LongitudinalEnv, load_pool, run_episode, sample_randomization

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "controllers" / "rl_longitudinal" / "policy.npz"
HANDMADE = [
    ROOT / "tracks" / "practice_basic.yaml",
    ROOT / "tracks" / "practice_technical.yaml",
    ROOT / "tracks" / "practice_obstacles.yaml",
    ROOT / "tracks" / "practice_loop.yaml",
    ROOT / "tracks" / "hackathon_test.yaml",
    ROOT / "tracks" / "brake_hairpin.yaml",
    ROOT / "tracks" / "accel_straights.yaml",
]


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def default_tracks() -> list:
    tracks = [p for p in HANDMADE if p.is_file()]
    train_dir = ROOT / "tracks" / "generated" / "train"
    if train_dir.is_dir():
        tracks.extend(sorted(train_dir.glob("*.yaml")))
    else:
        tracks.extend(sorted((ROOT / "tracks" / "generated").glob("*.yaml")))
    if not tracks:
        raise SystemExit("no course YAML found to train on")
    return tracks


def holdout_tracks() -> list:
    hold = ROOT / "tracks" / "generated" / "holdout"
    paths = sorted(hold.glob("*.yaml")) if hold.is_dir() else []
    extra = [ROOT / "tracks" / "practice_technical.yaml", ROOT / "tracks" / "hackathon_test.yaml"]
    return [p for p in extra + paths if p.is_file()]


# --------------------------------------------------------------------------- numpy export
def export_torch_actor(actor, obs_mean, obs_std, path: Path) -> None:
    torch = _torch()
    weights, biases = [], []
    for mod in actor.modules():
        if isinstance(mod, torch.nn.Linear):
            weights.append(mod.weight.detach().cpu().numpy().T)
            biases.append(mod.bias.detach().cpu().numpy())
    MLPPolicy(weights, biases, obs_mean, obs_std).save(path)


# --------------------------------------------------------------------------- torch models
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

    actor, critic = MLP(1).to(device), MLP(1).to(device)
    log_std = torch.nn.Parameter(torch.tensor(-1.4, device=device))
    return actor, critic, log_std


def _normalize(x, mean, std):
    return np.clip((x - mean) / std, -6.0, 6.0)


# --------------------------------------------------------------------------- workers
def _worker_loop(conn, seed, track_paths, max_time_s):
    env = LongitudinalEnv(randomize=True, max_time_s=max_time_s, seed=seed)
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
        obs, act, logp, rew, val, done = [], [], [], [], [], []
        ep_info = []
        running = 0.0
        for _ in range(n):
            feat = env._feat
            u_t = env.teacher_action()
            if policy is None:
                residual = 0.0
            else:
                residual = policy.act(feat)
                noise = float(rng.normal() * msg["std_action"])
                residual = float(np.clip(np.tanh(np.arctanh(np.clip(residual, -0.95, 0.95)) + noise), -1.0, 1.0))
            u = combine_u(u_t, residual)
            nxt, r, d, info = env.step(u)
            obs.append(feat)
            act.append(residual)
            logp.append(0.0)
            rew.append(r)
            val.append(0.0)
            done.append(float(d))
            running += r
            if d:
                ep_info.append(info)
                running = 0.0
                env.reset(track=tracks[int(rng.integers(0, len(tracks)))])
        conn.send({"obs": np.asarray(obs), "act": np.asarray(act), "rew": np.asarray(rew),
                   "done": np.asarray(done), "ep_info": ep_info, "unfinished": running})


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


# --------------------------------------------------------------------------- training
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


def imitate(env, tracks, actor, opt, torch, device, mean, std, steps=8000, epochs=6):
    rng = np.random.default_rng(0)
    obs_buf, tgt = [], []
    env.reset(track=str(tracks[0]), seed=1, rand=NOMINAL)
    for _ in range(steps):
        obs_buf.append(env._feat.copy())
        tgt.append(0.0)
        _, _, done, _ = env.step(env.teacher_action())
        if done:
            env.reset(track=str(tracks[int(rng.integers(0, len(tracks)))]), rand=NOMINAL)
    x = torch.tensor(_normalize(np.asarray(obs_buf), mean, std), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(tgt), dtype=torch.float32, device=device).unsqueeze(-1)
    print(f"warm-start: imitating the teacher on {len(tgt)} steps")
    for epoch in range(epochs):
        perm = torch.randperm(len(y), device=device)
        total, n = 0.0, 0
        for s in range(0, len(y), 256):
            b = perm[s:s + 256]
            if len(b) < 2:
                continue
            pred = actor(x[b])
            loss = torch.mean((pred - y[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"  imitate epoch {epoch + 1}/{epochs}  mse {total / max(n, 1):.4f}")


def validate(policy: MLPPolicy, tracks, seed=0) -> dict:
    env = LongitudinalEnv(randomize=False, seed=seed + 9)
    rows = [run_episode(env, lambda f, e: combine_u(e.teacher_action(), policy.act(f)), str(p), seed=seed, rand=NOMINAL) for p in tracks]
    fin = [r for r in rows if r["finished"]]
    times = [r["time"] / max(r["reference"], 1e-3) for r in fin]
    cones = sum(r["cones"] for r in rows)
    score = (100.0 * (len(fin) / max(len(rows), 1))
             - 8.0 * sum(1 for r in rows if r["dnf"])
             - 2.0 * cones
             - 20.0 * (float(np.mean(times)) - 1.0 if times else 1.0))
    return {"score": score, "finish": len(fin), "n": len(rows), "cones": cones,
            "ratio": float(np.mean(times)) if times else float("nan"), "rows": rows}


def train_torch(tracks, holdout, out, minutes, workers, device, hidden, seed, rollout, max_time_s, lr):
    torch = _torch()
    device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"torch {torch.__version__} device={device} workers={workers}")
    actor, critic, log_std = _build_models(torch, FEATURE_DIM, hidden, device)
    with torch.no_grad():
        last = list(actor.modules())[-1]
        if hasattr(last, "weight"):
            last.weight.zero_()
            last.bias.zero_()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=lr)
    mean, var, count = np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM), 1.0

    env = LongitudinalEnv(randomize=True, max_time_s=max_time_s, seed=seed)
    imitate(env, tracks, actor, torch.optim.Adam(actor.parameters(), lr=3e-3), torch, device,
            mean, np.sqrt(var), steps=8000)
    export_torch_actor(actor, mean, np.sqrt(var), out)
    print(f"wrote warm-start {out}")
    warm = MLPPolicy.load(out)
    best_score, best_state = -1e9, None
    val0 = validate(warm, holdout, seed=seed)
    print(f"  warm-start validate score {val0['score']:.1f}  finish {val0['finish']}/{val0['n']}  "
          f"cones {val0['cones']}  time/ref {val0['ratio']:.3f}")
    best_score = val0["score"]
    ws, bs, mu, sd = [np.array(w) for w in warm.weights], [np.array(b) for b in warm.biases], warm.obs_mean.copy(), warm.obs_std.copy()
    best_state = (ws, bs, mu, sd)

    procs = _spawn_workers(max(workers, 1), tracks, max_time_s, seed) if workers > 1 else None
    t0 = time.perf_counter()
    deadline = t0 + minutes * 60.0
    steps_done = 0
    update = 0
    rng = np.random.default_rng(seed)

    def current_npz_arrays():
        ws, bs = [], []
        for mod in actor.modules():
            if isinstance(mod, torch.nn.Linear):
                ws.append(mod.weight.detach().cpu().numpy().T)
                bs.append(mod.bias.detach().cpu().numpy())
        return ws, bs, mean, np.sqrt(var + 1e-8)

    while time.perf_counter() < deadline:
        ws, bs, mu, sd = current_npz_arrays()
        std_action = float(np.exp(log_std.detach().cpu().item()))
        if procs:
            _broadcast(procs, {"cmd": "set", "weights": ws, "biases": bs, "mean": mu, "std": sd})
            batch = _collect(procs, rollout // max(workers, 1), std_action)
            obs_np, act_np, rew_np, done_np = batch["obs"], batch["act"], batch["rew"], batch["done"]
            infos = batch["ep_info"]
        else:
            obs_np, act_np, rew_np, done_np, infos = [], [], [], [], []
            env.reset(track=str(tracks[int(rng.integers(0, len(tracks)))]))
            for _ in range(rollout):
                feat = env._feat.copy()
                loc = float(actor(torch.tensor(_normalize(feat, mu, sd), dtype=torch.float32, device=device)).item())
                u_pre = loc + std_action * float(rng.normal())
                u = combine_u(env.teacher_action(), float(np.tanh(u_pre)))
                _, r, d, info = env.step(u)
                obs_np.append(feat)
                act_np.append(u_pre)
                rew_np.append(r)
                done_np.append(float(d))
                if d:
                    infos.append(info)
                    env.reset(track=str(tracks[int(rng.integers(0, len(tracks)))]))
            obs_np, act_np, rew_np, done_np = map(np.asarray, (obs_np, act_np, rew_np, done_np))

        # features are already scaled to O(1); keep mean/std at 0/1 so the warm-start stays valid
        std = np.ones(FEATURE_DIM)

        x = torch.tensor(_normalize(obs_np, mean, std), dtype=torch.float32, device=device)
        with torch.no_grad():
            values = critic(x).squeeze(-1).cpu().numpy()
            loc0 = actor(x).squeeze(-1)
        last_v = 0.0 if done_np[-1] else float(values[-1])
        adv, ret = _gae(rew_np, values, done_np, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if procs:
            u_pre = torch.atanh(torch.tensor(np.clip(act_np, -0.999, 0.999), dtype=torch.float32, device=device))
        else:
            u_pre = torch.tensor(act_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            std0 = torch.exp(log_std)
            old_logp = (-0.5 * ((u_pre - loc0) / std0) ** 2 - log_std
                        - torch.log(1.0 - torch.tanh(u_pre) ** 2 + 1e-6))
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
                loc_b = actor(x[b]).squeeze(-1)
                std_t = torch.exp(log_std)
                logp = (-0.5 * ((u_pre[b] - loc_b) / std_t) ** 2 - log_std
                        - torch.log(1.0 - torch.tanh(u_pre[b]) ** 2 + 1e-6))
                ratio = torch.exp(torch.clamp(logp - old_logp[b], -20, 20))
                pol = -torch.mean(torch.minimum(ratio * adv_t[b], torch.clamp(ratio, 0.8, 1.2) * adv_t[b]))
                vf = 0.5 * torch.mean((critic(x[b]).squeeze(-1) - ret_t[b]) ** 2)
                ent = 0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std
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
        mean_ret = float(np.mean(rew_np) * 400) if len(rew_np) else 0.0
        elapsed = time.perf_counter() - t0
        print(f"  steps {steps_done:7d}  loss {last_loss:8.3f}  ep_return~{mean_ret:7.1f}  "
              f"episodes {len(infos):3d}  finish {finishes}  dnf {dnfs}  {elapsed:.0f}s")

        if update % 4 == 0 or time.perf_counter() + 30 > deadline:
            ws, bs, mu, sd = current_npz_arrays()
            pol = MLPPolicy(ws, bs, mu, sd)
            val = validate(pol, holdout, seed=seed)
            print(f"  validate score {val['score']:.1f}  finish {val['finish']}/{val['n']}  "
                  f"cones {val['cones']}  time/ref {val['ratio']:.3f}")
            if val["score"] >= best_score:
                best_score = val["score"]
                best_state = (copy.deepcopy(ws), copy.deepcopy(bs), mu.copy(), sd.copy())
                export_torch_actor(actor, mu, sd, out)
                print(f"  wrote {out} (best so far)")

    if procs:
        _broadcast(procs, {"cmd": "stop"})
        for p, _ in procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
    if best_state:
        MLPPolicy(*best_state).save(out)
    else:
        export_torch_actor(actor, mean, np.sqrt(var / max(count - 1.0, 1.0) + 1e-8), out)
    print(f"wrote {out}  best_validate={best_score:.1f}  steps={steps_done}")


def train_numpy_imitate(tracks, out, steps=12000):
    """No-torch fallback: fit the numpy MLP to the teacher."""
    print("torch not installed: fitting a numpy network to the teacher")
    env = LongitudinalEnv(randomize=False, seed=0)
    rng = np.random.default_rng(0)
    policy = MLPPolicy.random(FEATURE_DIM, (64, 64), rng)
    obs, tgt = [], []
    env.reset(track=str(tracks[0]), seed=0, rand=NOMINAL)
    for _ in range(steps):
        obs.append(env._feat.copy())
        tgt.append(0.0)
        _, _, done, _ = env.step(env.teacher_action())
        if done:
            env.reset(track=str(tracks[int(rng.integers(0, len(tracks)))]), rand=NOMINAL)
    x = np.asarray(obs)
    y = np.asarray(tgt)
    mean, std = x.mean(0), x.std(0) + 1e-6
    policy.obs_mean, policy.obs_std = mean, std
    # one-layer-at-a-time gradient on the last linear layer is enough as a fallback
    xn = np.clip((x - mean) / std, -6, 6)
    lr = 3e-3
    for epoch in range(12):
        h = xn
        for w, b in zip(policy.weights[:-1], policy.biases[:-1]):
            h = np.tanh(h @ w + b)
        pred = (h @ policy.weights[-1] + policy.biases[-1]).reshape(-1)
        err = pred - y
        policy.weights[-1] -= lr * (h.T @ err[:, None]) / len(y)
        policy.biases[-1] -= lr * err.mean()
        print(f"  imitate epoch {epoch + 1}/12  mse {float(np.mean(err ** 2)):.4f}")
    policy.save(out)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", nargs="*", help="course YAML files (default: handmade + generated/train)")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--max-time", type=float, default=150.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    tracks = [Path(p) for p in args.tracks] if args.tracks else default_tracks()
    missing = [str(p) for p in tracks if not Path(p).is_file()]
    if missing:
        raise SystemExit("missing courses:\n  " + "\n  ".join(missing))
    print("training on")
    for p in tracks:
        print(f"  {p}")
    hold = holdout_tracks()
    torch = _torch()
    if torch is None:
        train_numpy_imitate(tracks, Path(args.out))
    else:
        # Windows spawn needs the guard; workers imported from this module
        train_torch(tracks, hold, Path(args.out), args.minutes, args.workers, args.device,
                    args.hidden, args.seed, args.rollout, args.max_time, args.lr)


if __name__ == "__main__":
    mp.freeze_support()
    main()

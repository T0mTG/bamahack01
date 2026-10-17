# How to run the harness, the speed policy, and new courses

All commands assume the repo root (`ecocar-hackathon-sim`) and PowerShell. The bicycle backend does not need CARLA. Scored runs still use an unreleased layout, so anything you train has to drive from `obs.road` (the 40 m of course ahead), not from a memorized test track.

The submission is the folder `controllers/rl_longitudinal/`. It is self-contained: numpy only, no `rl/` package, no torch, no GPU. Training lives in `rl/` and writes `controllers/rl_longitudinal/policy.npz`.

---

## 1. One-time setup

Python 3.10–3.12 is what the harness asks for. 3.14 works for the bicycle backend (this machine). From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.venv\Scripts\Activate.ps1
```

If activation is blocked, call `.venv\Scripts\python.exe` instead of `python` in every command below.

That install is numpy, PyYAML, psutil, matplotlib, and pygame. Check it:

```powershell
python -m autocross check
python -m pytest
```

### GPU training venv (optional, E: drive)

C: is tight. Put a second venv on E: and install CUDA PyTorch there. The harness itself stays torch-free.

```powershell
python -m venv E:\ecocar_rl\venv
$env:TEMP="E:\ecocar_rl\tmp"; $env:TMP="E:\ecocar_rl\tmp"; $env:PIP_CACHE_DIR="E:\ecocar_rl\pipcache"
E:\ecocar_rl\venv\Scripts\python.exe -m pip install --upgrade pip
E:\ecocar_rl\venv\Scripts\python.exe -m pip install numpy PyYAML psutil matplotlib pytest
E:\ecocar_rl\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If the CUDA wheel is not available for your Python, drop `--index-url` and you get a CPU torch. Training still works; rollouts are the bottleneck, not the GPU.

Verify:

```powershell
E:\ecocar_rl\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Use that interpreter for `python -m rl.train`. Use the repo `.venv` (or system Python with numpy) for `python -m autocross run`. pygame is only needed for `--view`.

---

## 1b. Browser

Pick a controller and a map in the browser. The bicycle model runs on this machine, then the page replays the lap. A pair you have already driven is saved and replays immediately.

```powershell
python -m autocross web
```

Stop it with Ctrl+C. Add `--no-browser` if you only want the URL (`http://127.0.0.1:8765`).

---

## 2. Run the baseline

Pure pursuit plus a conservative speed loop. This is the thing to beat. `--view` opens a top-down window.

```powershell
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml --backend bicycle --view
```

Other public courses:

```powershell
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\practice_basic.yaml --backend bicycle
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\practice_technical.yaml --backend bicycle
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\practice_obstacles.yaml --backend bicycle
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\practice_loop.yaml --backend bicycle
```

A finished run prints a score and writes `runs\<timestamp>_<team>_<course>\`. Open `report.png` first. Exit code 0 = finish, 2 = DNF.

---

## 3. Run the longitudinal RL controller

`controllers\rl_longitudinal\controller.py`:

- Steering: pure pursuit on a path that swings around `road.obstacles`.
- Throttle / brake: analytic speed profile (the teacher) plus a small residual from a numpy network (`policy.npz`), wrapped by a safety shield (friction-circle throttle cap, forced brake above a grip-limit speed, never both pedals). A zero residual is exactly the teacher.
- Fallback: if `policy.npz` is missing, it drives the teacher alone (`mode: teacher`).

```powershell
python -m autocross run -c controllers\rl_longitudinal\controller.py -t tracks\hackathon_test.yaml --backend bicycle --view
```

Analytic teacher only (no network):

```powershell
python -m autocross run -c controllers\rl_longitudinal\controller.py -t tracks\hackathon_test.yaml --backend bicycle --controller-config '{"mode":"teacher"}'
```

`Command.debug` logs `dbg_u`, `dbg_v_soft`, `dbg_v_hard`, `dbg_shift` in `telemetry.csv`.

The weights sit next to the controller. The sandbox allows that read. It cannot read `tracks\` or `runs\`.

---

## 4. Courses

### Hand-written speed courses

| File | What it forces |
|---|---|
| `tracks\brake_hairpin.yaml` | Long straight into a 12 m hairpin. Accelerate, then brake. |
| `tracks\accel_straights.yaml` | Straights split by medium corners. Brake in, accelerate out. |

```powershell
python -m autocross show-track tracks\brake_hairpin.yaml
python -m autocross show-track tracks\accel_straights.yaml
```

### Random courses (built into the harness)

```powershell
python -m autocross gen-track --seed 42 --style technical --out tracks\my_random.yaml --preview
python -m autocross show-track tracks\my_random.yaml
python -m autocross run -c controllers\rl_longitudinal\controller.py -t tracks\my_random.yaml --backend bicycle --view
```

`--style` presets:

| Style | Layout |
|---|---|
| `fast` | long straights and sweepers |
| `technical` | kink, decreasing-radius chain, hairpin after a straight, esses, obstacle gates |
| `braking` | long straight into a hairpin |
| `accel` | straight / medium corner / straight |
| `obstacles` | mixed layout, dense unmapped cone groups |
| `mixed` | original random mix |
| `closed` | closed loop (same as `--closed`) |

Other flags: `--closed`, `--laps 2`, `--length 600`, `--width 6`, `--obstacles 4`, `--name my_course`. Omit `--seed` only for a secret course (the seed is not printed).

### The generated suite

```powershell
python scripts\generate_track_suite.py
```

Writes:

| Path | What |
|---|---|
| `tracks\generated\open_*.yaml`, `closed_loop.yaml` | four demo courses (same as before) |
| `tracks\generated\train\` | training pool + `manifest.json` |
| `tracks\generated\holdout\` | never used by training; for `rl.evaluate` |
| a PNG next to every YAML | preview |

Custom:

```powershell
python scripts\generate_track_suite.py --style technical --count 4 --seed-start 100 --out tracks\generated\train
```

`selftest` only picks up `tracks\*.yaml`, not `tracks\generated\`.

---

## 5. Train the speed policy

Training rolls the bicycle model at the same 40 Hz physics / 20 Hz control as a real run. Physics and perception are randomized per episode (grip, mass, brake/drive force, actuator delay, road-model noise, sight distance 25–40 m) so the policy does not overfit the bicycle model. Half the official score is CARLA.

```powershell
# GPU venv on E: (recommended)
E:\ecocar_rl\venv\Scripts\python.exe -m rl.train --minutes 20 --device cuda --out controllers\rl_longitudinal\policy.npz

# same machine, no torch: imitates the analytic teacher and writes a usable policy.npz
python -m rl.train
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--minutes` | 20 | wall-clock budget |
| `--workers` | cores-1 | CPU rollout processes |
| `--device` | cuda | `cuda` or `cpu` |
| `--rollout` | 2048 | steps per PPO update |
| `--tracks` | handmade + `generated/train` | course list |
| `--out` | `controllers/rl_longitudinal/policy.npz` | export path |
| `--seed` | 0 | |

What happens:

1. The actor is a *residual* on the analytic teacher (`u = teacher + 0.35 * tanh(net)`). It is initialized at zero, so the first checkpoint is the teacher.
2. PPO on mixed courses with domain randomization, small learning rate, so it can only nudge pedals.
3. Every few updates, score the holdout set. The best checkpoint is what gets saved.

Rollouts are the slow part (about 250 env steps/s per core). The GPU only does the tiny network update. 11 workers for 20 minutes is on the order of a few million control steps.

Fast in-process probe (no sandbox), for tuning the planner:

```powershell
python -m rl.probe
python -m rl.probe --planner "{\"lat_frac_soft\": 0.65}" --tracks hackathon_test practice_technical
python -m rl.probe --policy controllers\rl_longitudinal\policy.npz
```

### Reward (per 50 ms step, `rl/env.py`)

- `+0.05` per metre of progress, `-0.005` time cost
- `-0.08` × fraction of the step with rear slip > 6 deg
- `-0.05` × fraction with grip demand > 110 %
- `-0.03 |Δu|` pedal smoothness
- `-3` cone, `-6` off course, `-8` missed gate, `-40` DNF
- finish: `+30 * clip((2 - time/reference) / 1.05, 0, 1)` (same idea as the 30-point completion-time category)

Steering is not learned.

---

## 6. Evaluate

Official-harness comparison (RL vs baseline, public + holdout + perturbations):

```powershell
python -m rl.evaluate
python -m rl.evaluate --teacher --no-perturb
```

Then the real `selftest` (every public `tracks\*.yaml`, official settings):

```powershell
python -m autocross selftest -c controllers\rl_longitudinal\controller.py
```

Weight `practice_technical` more than the others when you estimate a score. The official technical course is 60% of the result and is a different shape.

Score one run, or add the judged innovation mark:

```powershell
python -m autocross score runs\<stamp>_rl_longitudinal_hackathon_test
python -m autocross score runs\<stamp>_rl_longitudinal_hackathon_test --judged innovation=80
python -m autocross leaderboard runs --award
python -m autocross plot runs\<stamp>_rl_longitudinal_hackathon_test
```

---

## 7. Why the eval course is unseen

`tracks\hackathon_test.yaml` is practice. The scored courses are not in this repo. `on_start` is given the name `"course"` and empty arrays. Plan from `obs.road`: curvature preview, corners, obstacles, finish distance when the finish is inside 40 m.

After training, try a seed you did not train on:

```powershell
python -m autocross gen-track --seed 99 --style technical --out tracks\holdout.yaml --preview
python -m autocross run -c controllers\rl_longitudinal\controller.py -t tracks\holdout.yaml --backend bicycle
```

---

## 8. Submit

Copy the whole folder. Do not include `rl/`.

```
teams/<your_name>/
  controller.py
  lr_planner.py
  lr_features.py
  lr_net.py
  policy.npz
```

The judges put that folder in `teams/` and run `selftest` plus the private evaluation courses on both backends. Before you hand it in:

```powershell
python -m autocross selftest -c controllers\rl_longitudinal\controller.py
```

---

## 9. Troubleshooting

| Symptom | What to do |
|---|---|
| PowerShell blocks `Activate.ps1` | Call `.venv\Scripts\python.exe` directly, or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `missing policy.npz` | The controller falls back to the teacher. Train with `python -m rl.train` to write the file. |
| `PermissionError` in the controller | You opened something under `tracks/` or `runs/`, or wrote a file. Read only files next to `controller.py`. Log with `Command.debug`. |
| Step overruns / p99 near 50 ms | The net is tiny; overruns usually mean you imported torch into the controller. Keep torch in `rl/` only. |
| `torch.cuda.is_available() == False` | Driver / wheel mismatch. Train with `--device cpu`. Rollouts dominate the time. |
| CUDA wheel fails on Python 3.14 | Use CPU torch, or a 3.12 venv on E:. |
| pygame fails to build on 3.14 | Skip it. `--view` needs pygame; headless `run` does not. |
| Training workers hang on Windows | `rl/train.py` uses `spawn`. If they hang, `--workers 1`. |
| DNF `stuck` / `lost` | The shield should brake on `road.valid == False`. If a generated course is degenerate, skip that seed. |
| Cone hits on obstacles | The planner shifts `preview_centerline`. Watch `dbg_shift`. Tighten `obstacle_margin` in `PlannerParams` and re-probe. |

### How CPU and GPU split the work

```
CPU workers (bicycle env, 20 Hz) --obs, reward--> GPU PPO update
                                          |
                                          +--> export policy.npz (numpy)
                                                    |
                                                    v
                              controller (1 core, no GPU, 50 ms deadline)
```

The event machine has 1 CPU core, no GPU, 1 GB RAM, and 50 ms per step. Inference is one 32→64→64→1 tanh map. That is well under the deadline (p99 is typically ~2 ms).

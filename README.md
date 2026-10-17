# mil-sim: EcoCAR Autocross Simulation Harness

A lightweight simulation harness for autonomous **path planning and control** on autocross courses. It is the EcoCAR hackathon simulator, and a starting point for model-in-the-loop (MIL) and software-in-the-loop (SIL) testing with ideal ground-truth data.

It follows the EIC autonomous autocross event: **the controller never gets a map.** Each control step it receives a road model of only the course visible ahead, and it must return steering, throttle, and brake inside an embedded-style compute budget. Teams do not write perception or sensor fusion.

Two backends drive the same rules and scoring:

- **Bicycle model** for fast iteration on a laptop, with no extra software. A 40 s lap takes about 1 s.
- **CARLA 0.9.16** for full physics. A 40 s lap takes about 3 s headless (tested on an RTX 30-series GPU).

The bicycle parameters are fitted to CARLA telemetry (`fit-bicycle`). `compare` checks lap-time agreement against a ~2.5% target. Official scoring counts each backend for half, so a team that only has the bicycle model is still scored on both.

Every run logs full ground-truth telemetry at 40 Hz. Command-by-command notes for the longitudinal RL controller and the track generator are in [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md).

---

## Quick start

### Requirements

| | Bicycle (development) | CARLA (evaluation) |
|---|---|---|
| OS | Windows, Linux, or macOS | Windows 10/11 or Linux |
| Python | **3.10–3.12** | same |
| Extra software | none | [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16) (~20 GB unpacked) |
| GPU | none | dedicated GPU |
| Packages | numpy, PyYAML, psutil, matplotlib, pygame, pytest | plus `carla==0.9.16` |

Students can develop and test entirely on the bicycle backend. The setup script installs the CARLA client when a wheel exists for the machine. On macOS there is no CARLA build, and the bicycle backend still works.

On Windows, install Python from python.org and tick **py launcher**. On macOS use `brew install python@3.12`. On Linux use the distribution `python3.12` and `python3.12-venv` packages.

### Setup

```powershell
git clone https://github.com/dszurek/ecocar-hackathon-sim.git
cd ecocar-hackathon-sim
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1      # Linux: ./scripts/setup.sh
.venv\Scripts\Activate.ps1                                      # Linux: source .venv/bin/activate
```

The script creates `.venv`, installs the harness, installs `carla==0.9.16` when it can, and runs a self-check. A missing CARLA client is expected on machines without it. If CARLA is not in `%USERPROFILE%\CARLA_0.9.16` or `C:\CARLA_0.9.16`, pass `-CarlaRoot <path>` or set `CARLA_ROOT`.

If PowerShell blocks activation, call `.venv\Scripts\python.exe` instead of `python`.

### Run something

```powershell
# Browser: pick a controller and a map, then watch the bicycle model
python -m autocross web

# Same idea in a desktop window
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml --backend bicycle --view

# Full CARLA (server first)
scripts\start_carla.ps1                  # windowed 1280x720, Low quality; -Headless for no window
python -m autocross check
python -m autocross run -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml --realtime
```

In CARLA the spectator follows the car, start and finish lines are painted on the pad, and the planned path is drawn in cyan. The pad is generated from a small OpenDRIVE road with knockable cones, so no Unreal map editing is required. Every run prints a score and writes a folder under `runs/`. Open `report.png` there first.

`python -m autocross web` serves `http://127.0.0.1:8765`. The first Drive for a controller and map simulates the lap. Later visits replay the saved result under `runs/web_cache/`. Opening `frontend/index.html` as a file does not work.

---

## Writing a controller

Copy `controllers/template_controller.py`. A controller is one Python file with one class that subclasses `autocross.api.Controller`:

```python
from autocross.api import Command, Controller

class MyController(Controller):
    def on_start(self, track, vehicle):
        super().on_start(track, vehicle)   # vehicle: mass, wheelbase, max_steer, mu, ...

    def step(self, obs):                    # called every 50 ms
        road = obs.road                     # the course visible ahead
        ...
        return Command(steer=s, throttle=t, brake=b,        # steer in [-1, 1], + = left
                       path=my_path, path_frame="vehicle")
```

### What each step receives

| Input | Contents |
|---|---|
| `on_start(track, vehicle)` | In event mode `track` arrays are empty and the course name is `"course"`. `vehicle` publishes the limits: 1,845 kg, 3.0 m wheelbase, 35° max steer, a steering-rate limit, μ ≈ 0.95, 5.2 m/s² acceleration, 9 m/s² braking, and the control period. |
| `obs.road` | The course up to 40 m ahead, vehicle frame (x forward, y left): lateral offset, heading error, curvature, distance to each boundary, a preview of centerline and boundaries, upcoming corners, obstacles, and finish distance once the finish is in range. |
| Vehicle state | Ground truth: `obs.x, y, yaw, vx, vy, yaw_rate, ax, ay, speed`, plus the actuator values actually applied. |
| `obs.cones` | Optional raw cone detections (vehicle frame, colour class) within 30 m and a 140° field of view. |

Definitions live in [`autocross/api.py`](autocross/api.py). [`controllers/baseline_pure_pursuit.py`](controllers/baseline_pure_pursuit.py) is a complete road-model controller: pure pursuit on the perceived centerline, speed from the curvature preview, and a throttle limit from leftover cornering grip. It ignores obstacles, finishes cleanly and slowly, and is the score to beat.

### Rules the harness enforces

- **Compute.** `step()` must return within 50 ms. A late step holds the previous command and logs an overrun. The budget is 1 CPU core, no GPU, and 1 GB of memory. `on_start()` may take up to 60 s. Timing includes background threads in the controller process.
- **Actuators.** Steering is slew-rate limited. Actuation delay can be added.
- **Invalid output.** An exception, NaN, or wrong return type is an error. Twenty errors in a row is a DNF. Scored error penalties count at most ten, at −1 each.
- **Dependencies.** The controller runs in the harness environment. numpy is always available. Use only packages installed there.
- **No course files.** The controller process cannot open anything under `tracks/` or `runs/`. Read your own lookup tables and weights from the controller folder.
- **Sandbox.** The process may read files next to the controller. It may not write, delete, or rename files, start processes, open network connections, import the CARLA client, call native code through `ctypes`, or inspect the harness through the garbage collector. Those raise `PermissionError` and count as controller errors. Log with `Command.debug`; the values show up as `dbg_*` columns in `telemetry.csv`.

Check a controller before you submit:

```powershell
python -m autocross selftest -c my_controller.py            # every public course, bicycle model
python -m autocross selftest -c my_controller.py --carla    # and CARLA, if the server is running
```

### Submitting

Submit one file, `<team>.py`, or a folder `<team>/` with `controller.py` plus helper modules and data files. Judges place it in `teams/` and run the official evaluation.

---

## Scoring

Rules are evaluated from ground truth at every physics tick, so both backends see the same events.

| Penalty | Points |
|---|---|
| Cone hit (each cone once) | −2 |
| Off course (all four wheels outside the cones, once per excursion) | −5 |
| Missed gate | −10 |
| Controller error (capped at 10) | −1 |
| Peak memory over the 1 GB ceiling | −5 |

A run is a **DNF** if the car stays off course for more than 5 s, gets more than 15 m from the course, drives the wrong way, is stuck for 8 s, rolls over, times out at 240 s, or hits the consecutive-error limit.

| Category | Weight | What it measures |
|---|---|---|
| Completion time | 30 | Lap time versus the course `reference_time`, or an automatic friction-limited centerline estimate |
| Path deviation | 15 | Mean and peak distance from the path published 0.5 s earlier, near the point the car should have reached. A missing or implausible path is scored against the centerline |
| Compute | 15 | p99 step time versus the deadline, overruns, and peak memory |
| Physical realism | 15 | Grip over-demand, rear-axle sliding, steering-rate saturation, and pedal overlap |
| Innovation | 25 | Judged. Enter it with `score --judged innovation=<0-100>` |

Publish a real plan in `Command.path` that reaches at least a few metres ahead. Each pose is compared with the path from 0.5 s earlier, so a stub just in front of the car scores poorly. Paths that fold back or zig-zag tighter than the car can drive are ignored.

The four computed categories are the run's **technical points** (out of 75), then penalties are subtracted (floor at 0). Weights and thresholds are in [`config/scoring.yaml`](config/scoring.yaml).

### Official evaluation

Judges run every submission on two unreleased courses, each on both backends: four runs per team, one batch, one reference machine. The 50 ms deadline depends on that machine.

| Course | Weight | Backend | Weight |
|---|---|---|---|
| Main course | 40% | CARLA (headless) | 50% |
| Technical course | 60% | Bicycle model | 50% |

**Final score = weighted average of the four technical totals (out of 75) + judged innovation (out of 25).**

A DNF or a crashed controller still counts, with the points of that run. The technical course is weighted higher, so a controller that is fast on the public courses and fails the technical elements loses more than one that is slower and clean. Award categories (fastest, most physically realistic, most efficient compute, most precise) use the same weights and require a finish on all four runs.

`python -m autocross selftest -c my_controller.py` is the local estimate. Weight `practice_technical` more heavily than the other public courses. Add `--carla` when the server is available.

---

## Courses

Courses are YAML. They are built from segments (`straight`, `arc`, `slalom`) or from explicit centerline points. Cones, gates, and start/finish lines are generated. `obstacles` are cone groups that are absent from any map and appear only in perception.

```yaml
name: practice_basic
closed: false            # point-to-point; closed: true and laps: N for a loop
width: 5.5
segments:
  - {type: straight, length: 35}
  - {type: arc, radius: 15, angle: -110}      # + = left
  - {type: slalom, cycles: 3, wavelength: 26, swing: 2.5}
obstacles:
  - {s: 120, offset: 1.0, width: 2.0}
```

| Course | What it is |
|---|---|
| `tracks/hackathon_test.yaml` | Main public test course |
| `tracks/practice_basic.yaml` | Short open course: corners, a slalom, a hairpin |
| `tracks/practice_obstacles.yaml` | Open course with unmapped obstacle groups |
| `tracks/practice_loop.yaml` | Closed course, 2 laps |
| `tracks/practice_technical.yaml` | Mild technical course: narrow kink, braking zone into a hairpin, decreasing radius, esses, obstacle gates |
| `tracks/brake_hairpin.yaml`, `tracks/accel_straights.yaml` | Short courses that force a brake or an accelerate-out |

Scored runs use **unreleased evaluation courses** that only the judges have: a main course with the same kinds of elements and width as `hackathon_test` but a different layout, and a harder technical course. `practice_technical` is the public stand-in for that second course. A controller tuned to the exact shape of the public tracks will not carry over. Plan from `obs.road`, including obstacles.

Generate extra courses before the event:

```powershell
python -m autocross gen-track --seed 42 --style technical --obstacles 3 --out tracks\my_random.yaml --preview
```

`--style` is `fast`, `technical`, `braking`, `accel`, `obstacles`, `mixed`, or `closed`. `scripts/generate_track_suite.py` writes the training and holdout pools under `tracks/generated/`. `selftest` only loads `tracks/*.yaml`.

**Judges.** Evaluation courses, `tracks/private/tools/evaluate.py`, and the reference checks live in a separate private repository, cloned into `tracks/private/` (git-ignored). Team controllers cannot read anything under `tracks/`. Put submissions in `teams/` as `<team>.py` or `<team>/controller.py` (also git-ignored). Omit `--seed` when generating a secret course: the seed is random and is neither printed nor saved. A small or guessable seed can be regenerated by anyone.

---

## Commands

`python -m autocross <command> --help` lists flags.

| Command | Purpose |
|---|---|
| `run -c CTRL -t TRACK` | One run. `--backend bicycle`, `--view`, `--realtime [FACTOR]`, `--no-render`, `--team NAME`, `--controller-config cfg.yaml`, `--perturb KEY=VALUE`, `--set section.key=value` |
| `web` | Browser UI on port 8765. `--no-browser` prints the URL only |
| `batch -c CTRL` | Robustness sweep: courses × perturbations × seeds ([`config/robustness.yaml`](config/robustness.yaml)) |
| `score RUN_DIR` | Re-score a run. `--judged innovation=80` adds the judged mark |
| `leaderboard [runs] --award` | Best run per team and course, plus award categories |
| `gen-track --out F.yaml` | Random valid course. `--seed`, `--style`, `--closed`, `--obstacles`, `--preview` |
| `show-track F.yaml` | Validate, print stats and reference time, save a preview PNG |
| `plot RUN_DIR` | Regenerate `report.png` |
| `selftest -c CTRL [--carla]` | Every public course with official settings, plus the sandbox checks |
| `check` | Install and CARLA connectivity |
| `calibrate --backend carla` | Measure acceleration, braking, and lateral grip |
| `compare -c CTRL -t TRACK` | Same controller on CARLA and the bicycle model. Exits 1 if any course exceeds `--tolerance` (default 2.5%). `--carla-runs DIR` reuses earlier CARLA runs |
| `fit-bicycle RUN_DIR... [--write]` | Fit bicycle parameters to CARLA telemetry. `--write` stores them in `config/sim.yaml` |

### Run folder (`runs/<timestamp>_<team>_<course>/`)

- `telemetry.csv` — pose, body velocities and accelerations, steering, progress, lateral offset, boundary margin, path error, commanded versus applied actuation, step timing, and `dbg_*` values
- `events.json` — start, finish, cone hits, off-course excursions, missed gates, DNF
- `summary.json` — metrics, score, vehicle parameters, config snapshot, controller and course hashes, machine info
- `report.png` — course map with the speed-coloured trajectory and events, plus speed, error, actuation, and compute plots

Exit code 0 is a finish. Exit code 2 is a DNF.

---

## Configuration

[`config/sim.yaml`](config/sim.yaml) is the official event setup. Override any value for one run with `--set section.key=value`.

| Section | Contents |
|---|---|
| `sim` | Physics step (25 ms), control rate (20 Hz), timeout, seed, realtime factor |
| `carla` | Host and port, rendering, spectator, pad, painted lines, cone models, weather, recorder |
| `vehicle` | Blueprint, steering limit, CARLA tire and brake settings, published limits, bicycle parameters |
| `road_model` | Preview range and step, corner detection, lane-model noise (0 = ideal) |
| `perception` | Cone sensor range, field of view, noise, dropout, misclassification |
| `track_knowledge` | `perception` (event: no map) or `full` (map given in `on_start`, for debugging) |
| `compute_budget` | Enforce or measure, cores, deadline, memory, GPU |
| `sandbox` | Paths the controller may not read (`tracks` and `runs` by default) |
| `rules` | Off-course rule, DNF thresholds, gate spacing, cone-hit margin |

CARLA's stock Model 3 is retuned for this event: tire friction is reduced to about 0.95 g lateral, brake torque is raised to about 9 m/s², drive torque is scaled to 75% so mid-corner throttle does not spin the tires, and max road-wheel steer is 35° instead of 70°. After a vehicle change, run `python -m autocross calibrate --backend carla` and update the published limits.

To refresh the bicycle fit on the reference machine, with CARLA running:

```powershell
python -m autocross compare -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml -t tracks\practice_loop.yaml -t tracks\practice_obstacles.yaml -t tracks\practice_basic.yaml
python -m autocross fit-bicycle runs\<stamp>_compare --write
python -m autocross compare -c controllers\baseline_pure_pursuit.py -t tracks\hackathon_test.yaml -t tracks\practice_loop.yaml -t tracks\practice_obstacles.yaml -t tracks\practice_basic.yaml --carla-runs runs\<stamp>_compare
```

The fitter replays logged CARLA runs in 1.5 s windows and matches body velocities, yaw rate, and wheel angle. Mass, wheelbase, and CG position are copied from the CARLA vehicle, so both backends publish the same `VehicleParams`. Check the fit on a course and a controller it did not see, then commit `config/sim.yaml`.

---

## Toward MIL / SIL

The `Controller` interface is the plug-in point. The simulator, rules, logging, and scoring stay the same.

- **MIL.** Wrap a Simulink model exported as an FMU (for example with `fmpy`) or called through the MATLAB Engine. Set `compute_budget.mode: measure`, because the loop is lockstep.
- **SIL.** Compile Embedded Coder output to a shared library and call its step function. Keep `mode: enforce` with ECU-like `cpu_cores` and `step_deadline_ms`. That approximates ECU timing. True timing needs PIL or HIL.
- **Logs.** Telemetry is a ground-truth reference for vehicle logs. `carla.record: true` also saves a CARLA replay.

Camera and lidar sensor models are not implemented. Perception is derived from ground truth, with configurable noise. There is no asynchronous real-time loop for HIL yet.

---

## Repository layout

```
autocross/            harness
  api.py              Controller, Observation, RoadModel, Command
  runner.py           closed loop, actuators, logging, batch sweeps
  sandbox.py          controller process and compute budget
  perception.py       road model and cone sensor
  monitor.py          race rules and events
  metrics.py score.py plot.py viewer.py web.py
  track.py trackgen.py geometry.py
  backends/           carla_backend.py, bicycle.py
  cli.py calibrate.py fit.py config.py
config/               sim.yaml, scoring.yaml, robustness.yaml
controllers/          baseline, template, rl_longitudinal/
rl/                   training only (not part of a submission)
tracks/               public courses; generated/; private/ is git-ignored
frontend/             browser UI served by `autocross web`
scripts/              setup, CARLA launch, generate_track_suite.py
docs/HOW_TO_RUN.md    setup, RL controller, training, evaluation
tests/                pytest (bicycle backend, no CARLA)
```

## Development

```powershell
python -m pytest
```

The suite runs on the bicycle backend and includes the sandbox checks. Run the official evaluation on one reference machine, because the 50 ms deadline is machine-dependent.

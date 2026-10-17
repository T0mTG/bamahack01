const canvas = document.querySelector("#track");
const ctx = canvas.getContext("2d");
const clockEl = document.querySelector("#clock");
const speedEl = document.querySelector("#speed");
const alongEl = document.querySelector("#along");
const bar = document.querySelector("#bar");
const barFill = document.querySelector("#bar-fill");
const finishLabel = document.querySelector("#finish-label");
const driveBtn = document.querySelector("#drive-btn");
const againBtn = document.querySelector("#again-btn");
const gridBtn = document.querySelector("#grid-btn");
const resultEl = document.querySelector("#result");
const banner = document.querySelector("#banner");
const mark = document.querySelector("#mark");
const live = document.querySelector("#live");
const viewButtons = document.querySelectorAll("[data-view]");
const rateButtons = document.querySelectorAll("[data-rate]");

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const CAT_LABELS = {
  completion_time: "Time",
  deviation: "Path",
  compute: "Compute",
  dynamics: "Dynamics",
};

const state = {
  catalog: null,
  controllerId: null,
  trackId: null,
  course: null,
  replay: null,
  t: 0,
  playing: false,
  finished: false,
  view: "map",
  grid: true,
  rate: 1,
  busy: false,
  runToken: 0,
  courseToken: 0,
};

function say(text) {
  live.textContent = text;
}

function key() {
  return `${state.controllerId}|${state.trackId}`;
}

function controller() {
  return state.catalog?.controllers.find((item) => item.id === state.controllerId) || null;
}

function trackMeta() {
  return state.catalog?.tracks.find((item) => item.id === state.trackId) || null;
}

function lerp(a, b, u) {
  return a + (b - a) * u;
}

function lerpAngle(a, b, u) {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return a + d * u;
}

function poseAt(t) {
  const drive = state.replay?.drive;
  const course = state.course;
  if (!drive || drive.x.length < 2) {
    const spawn = course?.spawn || { x: 0, y: 0, yaw: 0 };
    return { x: spawn.x, y: spawn.y, yaw: spawn.yaw, v: 0, s: 0, steer: 0, index: 0 };
  }
  const end = (drive.x.length - 1) * drive.dt;
  const clamped = Math.max(0, Math.min(end, t));
  const f = clamped / drive.dt;
  const i = Math.min(drive.x.length - 2, Math.floor(f));
  const u = f - i;
  return {
    x: lerp(drive.x[i], drive.x[i + 1], u),
    y: lerp(drive.y[i], drive.y[i + 1], u),
    yaw: lerpAngle(drive.yaw[i], drive.yaw[i + 1], u),
    v: lerp(drive.v[i], drive.v[i + 1], u),
    s: lerp(drive.s[i], drive.s[i + 1], u),
    steer: lerpAngle(drive.steer[i], drive.steer[i + 1], u),
    index: i,
  };
}

function endTime() {
  const drive = state.replay?.drive;
  if (!drive || drive.x.length < 2) return 0;
  return (drive.x.length - 1) * drive.dt;
}

function clockText(t) {
  const s = Math.max(0, Math.floor(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.offsetWidth;
  const cssH = canvas.offsetHeight;
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { dpr, w: cssW, h: cssH };
}

function worldMatrix(dpr, w, h, pose) {
  const course = state.course;
  if (state.view === "map") {
    const [x0, y0, x1, y1] = course.bounds;
    const pad = 28;
    const ppm = Math.min((w - pad * 2) / Math.max(x1 - x0, 1), (h - pad * 2) / Math.max(y1 - y0, 1));
    const ox = (w - (x1 - x0) * ppm) / 2 - x0 * ppm;
    const oy = (h + (y1 - y0) * ppm) / 2 + y0 * ppm;
    ctx.setTransform(dpr * ppm, 0, 0, dpr * -ppm, dpr * ox, dpr * oy);
    return ppm;
  }
  const ppm = h / 34;
  const c = Math.cos(pose.yaw);
  const s = Math.sin(pose.yaw);
  const m11 = ppm * s;
  const m12 = -ppm * c;
  const m21 = -ppm * c;
  const m22 = -ppm * s;
  const ox = w / 2 - m11 * pose.x - m12 * pose.y;
  const oy = h * 0.62 - m21 * pose.x - m22 * pose.y;
  ctx.setTransform(dpr * m11, dpr * m21, dpr * m12, dpr * m22, dpr * ox, dpr * oy);
  return ppm;
}

function strokePoly(pts) {
  if (!pts || pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
}

function drawGrid(pose, ppm) {
  const course = state.course;
  ctx.save();
  ctx.strokeStyle = "#e6e6e6";
  ctx.lineWidth = 1 / ppm;
  const reach = state.view === "map" ? 0 : 60;
  let x0;
  let y0;
  let x1;
  let y1;
  if (state.view === "map") {
    [x0, y0, x1, y1] = course.bounds;
  } else {
    x0 = pose.x - reach;
    x1 = pose.x + reach;
    y0 = pose.y - reach;
    y1 = pose.y + reach;
  }
  const g = 10;
  ctx.beginPath();
  for (let x = Math.floor(x0 / g) * g; x <= x1; x += g) {
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
  }
  for (let y = Math.floor(y0 / g) * g; y <= y1; y += g) {
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawRoad(ppm) {
  const course = state.course;
  const left = course.left;
  const right = course.right;
  if (!left.length || !right.length) return;
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (let i = 1; i < left.length; i += 1) ctx.lineTo(left[i][0], left[i][1]);
  for (let i = right.length - 1; i >= 0; i -= 1) ctx.lineTo(right[i][0], right[i][1]);
  ctx.closePath();
  ctx.fillStyle = "#f7f7f7";
  ctx.fill();

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2.1 / ppm;
  strokePoly(left);
  strokePoly(right);

  ctx.save();
  ctx.strokeStyle = "#bdbdbd";
  ctx.lineWidth = 1.2 / ppm;
  ctx.setLineDash([1.4, 1.1]);
  strokePoly(course.center);
  ctx.restore();

  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2.6 / ppm;
  strokePoly(course.startLine);
  ctx.lineWidth = 2.2 / ppm;
  strokePoly(course.finishLine);
}

function drawCones(ppm) {
  const minR = (state.view === "map" ? 0.85 : 1.7) / ppm;
  for (const cone of state.course.cones) {
    const [x, y, color, mapped] = cone;
    const r = Math.max(color === 2 && mapped === 0 ? 0.42 : 0.34, minR);
    ctx.beginPath();
    if (color === 2 && mapped === 0) {
      ctx.moveTo(x, y + r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y - r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      ctx.fillStyle = "#111";
      ctx.fill();
      continue;
    }
    ctx.arc(x, y, color === 2 ? 0.46 : r, 0, Math.PI * 2);
    if (color === 0) {
      ctx.fillStyle = "#111";
      ctx.fill();
    } else {
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.lineWidth = 1.5 / ppm;
      ctx.strokeStyle = "#111";
      ctx.stroke();
    }
  }
}

function drawTrail(pose, ppm) {
  const drive = state.replay?.drive;
  if (!drive) return;
  const last = pose.index;
  if (last < 2) return;
  const step = state.view === "map" ? 3 : 2;
  ctx.beginPath();
  ctx.moveTo(drive.x[0], drive.y[0]);
  for (let i = step; i <= last; i += step) ctx.lineTo(drive.x[i], drive.y[i]);
  ctx.lineTo(pose.x, pose.y);
  ctx.strokeStyle = "#111";
  ctx.lineWidth = (state.view === "trail" ? 2.8 : 1.8) / ppm;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

function drawEvents(ppm, t) {
  const events = state.replay?.events || [];
  for (const event of events) {
    if (event.t > t + 0.02) continue;
    const r = (state.view === "map" ? 1.3 : 0.7) ;
    ctx.beginPath();
    if (event.kind === "cone_hit") {
      ctx.arc(event.x, event.y, Math.max(r, 2.2 / ppm), 0, Math.PI * 2);
      ctx.fillStyle = "#111";
      ctx.fill();
    } else {
      ctx.arc(event.x, event.y, Math.max(r * 1.4, 3 / ppm), 0, Math.PI * 2);
      ctx.lineWidth = 1.6 / ppm;
      ctx.strokeStyle = "#111";
      ctx.stroke();
    }
  }
}

function vehiclePoint(pose, vx, vy) {
  const c = Math.cos(pose.yaw);
  const s = Math.sin(pose.yaw);
  return [pose.x + vx * c - vy * s, pose.y + vx * s + vy * c];
}

function drawWheel(pose, vx, vy, steer, scale) {
  const c = Math.cos(pose.yaw + steer);
  const s = Math.sin(pose.yaw + steer);
  const [cx, cy] = vehiclePoint(pose, vx, vy);
  const hw = 0.34 * scale;
  const hh = 0.16 * scale;
  const corners = [[hw, hh], [hw, -hh], [-hw, -hh], [-hw, hh]];
  ctx.beginPath();
  corners.forEach(([px, py], i) => {
    const x = cx + px * c - py * s;
    const y = cy + px * s + py * c;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "#111";
  ctx.fill();
}

function drawCar(pose, ppm) {
  const veh = state.course.vehicle;
  const scale = Math.max(1, 18 / ppm / veh.length);
  const half = (veh.wheelbase / 2) * scale;
  const track = 0.82 * scale;
  drawWheel(pose, half, track, pose.steer, scale);
  drawWheel(pose, half, -track, pose.steer, scale);
  drawWheel(pose, -half, track, 0, scale);
  drawWheel(pose, -half, -track, 0, scale);

  const L = veh.length * scale;
  const W = veh.width * scale;
  const nose = L * 0.5;
  const tail = -L * 0.5;
  const body = [
    [nose - 0.35, W * 0.5 - 0.28],
    [nose, W * 0.28],
    [nose, -W * 0.28],
    [nose - 0.35, -W * 0.5 + 0.28],
    [tail + 0.15, -W * 0.5 + 0.22],
    [tail, -W * 0.32],
    [tail, W * 0.32],
    [tail + 0.15, W * 0.5 - 0.22],
  ];
  ctx.beginPath();
  body.forEach(([vx, vy], i) => {
    const [x, y] = vehiclePoint(pose, vx, vy);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2.3 / ppm;
  ctx.lineJoin = "round";
  ctx.fill();
  ctx.stroke();
}

function renderHud(pose) {
  const course = state.course;
  const finish = course ? course.finish : 1;
  const kmh = Math.round(pose.v * 3.6);
  const along = Math.max(0, Math.round(pose.s));
  const pct = Math.max(0, Math.min(1, pose.s / finish));
  speedEl.textContent = String(kmh);
  alongEl.textContent = pose.s >= finish - 0.5 && state.replay ? "finish" : `${along} m`;
  barFill.style.width = `${pct * 100}%`;
  bar.setAttribute("aria-valuemax", String(Math.round(finish)));
  bar.setAttribute("aria-valuenow", String(Math.min(along, Math.round(finish))));
  finishLabel.textContent = `${Math.round(finish)} m`;
  clockEl.textContent = clockText(state.t);
  const ctrl = controller();
  const map = trackMeta();
  if (ctrl && map) mark.textContent = `${ctrl.name} · ${map.name}`;
  const matched = state.replay && state.replay.key === key();
  if (state.busy) driveBtn.textContent = "Simulating";
  else if (!matched) driveBtn.textContent = "Drive";
  else if (state.finished) driveBtn.textContent = "Replay";
  else if (state.playing) driveBtn.textContent = "Pause";
  else driveBtn.textContent = "Drive";
  driveBtn.disabled = state.busy;
  againBtn.hidden = !matched || state.busy;
}

function paint() {
  const { dpr, w, h } = resize();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.course) {
    ctx.fillStyle = "#111";
    ctx.font = "600 16px Outfit, Segoe UI, sans-serif";
    ctx.fillText(state.catalog ? "Loading map…" : "Connecting…", 24, 40);
    return;
  }
  const pose = poseAt(state.t);
  const ppm = worldMatrix(dpr, w, h, pose);
  if (state.grid) drawGrid(pose, ppm);
  drawRoad(ppm);
  const drive = state.replay?.drive;
  if (state.view !== "chase") drawTrail(pose, ppm);
  else if (drive && pose.index > 8) {
    ctx.beginPath();
    const from = Math.max(0, pose.index - 28);
    ctx.moveTo(drive.x[from], drive.y[from]);
    for (let i = from + 2; i <= pose.index; i += 2) ctx.lineTo(drive.x[i], drive.y[i]);
    ctx.lineTo(pose.x, pose.y);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 1.6 / ppm;
    ctx.lineCap = "round";
    ctx.stroke();
  }
  drawCones(ppm);
  drawEvents(ppm, state.t);
  drawCar(pose, ppm);
  renderHud(pose);
}

function choiceButton(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "choice";
  button.role = "radio";
  button.dataset.id = item.id;
  button.innerHTML = `<strong></strong><span></span>`;
  button.querySelector("strong").textContent = item.name;
  button.querySelector("span").textContent = item.blurb;
  return button;
}

function syncPressed() {
  document.querySelectorAll("#controllers .choice").forEach((button) => {
    button.setAttribute("aria-checked", button.dataset.id === state.controllerId ? "true" : "false");
  });
  document.querySelectorAll("#maps .choice").forEach((button) => {
    button.setAttribute("aria-checked", button.dataset.id === state.trackId ? "true" : "false");
  });
}

function renderCatalog() {
  const controllers = document.querySelector("#controllers");
  const maps = document.querySelector("#maps");
  controllers.replaceChildren();
  maps.replaceChildren();
  for (const item of state.catalog.controllers) {
    const button = choiceButton(item);
    button.addEventListener("click", () => selectController(item.id));
    controllers.append(button);
  }
  let group = "";
  for (const item of state.catalog.tracks) {
    if (item.group !== group) {
      group = item.group;
      const label = document.createElement("p");
      label.className = "group-label";
      label.textContent = group;
      maps.append(label);
    }
    const button = choiceButton(item);
    button.addEventListener("click", () => selectTrack(item.id));
    maps.append(button);
  }
  syncPressed();
}

function clearReplay() {
  state.replay = null;
  state.t = 0;
  state.playing = false;
  state.finished = false;
  resultEl.innerHTML = `<h2>Result</h2><p class="result-empty" id="result-empty">Press Drive to run ${controller()?.name || "the controller"} on ${trackMeta()?.name || "this map"}.</p>`;
  againBtn.hidden = true;
}

function selectController(id) {
  if (id === state.controllerId) return;
  state.runToken += 1;
  state.busy = false;
  banner.hidden = true;
  state.controllerId = id;
  syncPressed();
  clearReplay();
  say(`${controller().name} selected.`);
}

function selectTrack(id) {
  if (id === state.trackId) return;
  state.runToken += 1;
  state.busy = false;
  banner.hidden = true;
  state.trackId = id;
  syncPressed();
  clearReplay();
  loadCourse(id);
}

async function loadCourse(id) {
  const token = ++state.courseToken;
  try {
    const res = await fetch(`/api/course?track=${encodeURIComponent(id)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not load the map.");
    if (token !== state.courseToken) return;
    state.course = data;
    canvas.setAttribute("aria-label", `Top-down view of ${data.name}.`);
    say(`${data.name} loaded.`);
  } catch (error) {
    if (token !== state.courseToken) return;
    showError(error.message);
  }
}

function showError(message) {
  resultEl.innerHTML = `<h2>Result</h2><p class="error"></p>`;
  resultEl.querySelector(".error").textContent = message;
  say(message);
}

function renderScore(score, cached) {
  const status = score.finished
    ? `Finished ${score.time_s.toFixed(1)} s`
    : `DNF · ${(score.dnf || "stopped").replaceAll("_", " ")}`;
  const progress = score.finished ? "" : ` · ${Math.round(score.progress * 100)}% of the course`;
  const ref = score.reference_time_s ? `Reference ${score.reference_time_s.toFixed(1)} s` : "";
  const cats = Object.entries(score.categories || {}).map(([name, value]) => {
    const label = CAT_LABELS[name] || name;
    return `<div class="cat"><span>${label}</span><b>${value.toFixed(1)}</b></div>`;
  }).join("");
  resultEl.innerHTML = `
    <h2>Result</h2>
    <p class="result-status"></p>
    <p class="result-sub"></p>
    <div class="cats">${cats}</div>
    <p class="penalties"></p>`;
  resultEl.querySelector(".result-status").textContent = status;
  resultEl.querySelector(".result-sub").textContent = [
    `Technical ${score.technical.toFixed(1)} / ${score.technical_max}`,
    ref,
    cached ? "Saved lap" : "Just simulated",
  ].filter(Boolean).join(" · ");
  resultEl.querySelector(".penalties").textContent =
    `Cones ${score.cones} · off course ${score.off_course} · missed gates ${score.missed_gates}${progress}`;
}

async function simulate(force) {
  const token = ++state.runToken;
  const ctrl = controller();
  const map = trackMeta();
  state.busy = true;
  state.playing = false;
  banner.hidden = false;
  banner.textContent = `Simulating ${ctrl.name} on ${map.name}…`;
  say(banner.textContent);
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ controller: state.controllerId, track: state.trackId, force }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "The run failed.");
    if (token !== state.runToken) return;
    state.course = data.course;
    state.replay = {
      drive: data.drive,
      events: data.events || [],
      score: data.score,
      key: key(),
      cached: data.cached,
    };
    state.t = 0;
    state.finished = false;
    state.playing = !reduced;
    renderScore(data.score, data.cached);
    const done = data.score.finished
      ? `${ctrl.name} finished ${map.name} in ${data.score.time_s.toFixed(1)} seconds.`
      : `${ctrl.name} did not finish ${map.name}.`;
    say(done);
  } catch (error) {
    if (token !== state.runToken) return;
    showError(error.message);
  } finally {
    if (token === state.runToken) {
      state.busy = false;
      banner.hidden = true;
    }
  }
}

function togglePlay() {
  if (!state.replay || state.replay.key !== key()) {
    simulate(false);
    return;
  }
  if (state.finished || state.t >= endTime() - 0.02) {
    state.t = 0;
    state.finished = false;
    state.playing = true;
    say("Replay started.");
    return;
  }
  state.playing = !state.playing;
  say(state.playing ? "Driving." : "Paused.");
}

driveBtn.addEventListener("click", togglePlay);
againBtn.addEventListener("click", () => simulate(true));

gridBtn.addEventListener("click", () => {
  state.grid = !state.grid;
  gridBtn.setAttribute("aria-pressed", state.grid ? "true" : "false");
});

viewButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    viewButtons.forEach((other) => {
      other.setAttribute("aria-pressed", other === button ? "true" : "false");
    });
  });
});

rateButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.rate = Number(button.dataset.rate);
    rateButtons.forEach((other) => {
      other.setAttribute("aria-pressed", other === button ? "true" : "false");
    });
  });
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) return;
  if (event.target instanceof HTMLButtonElement) return;
  event.preventDefault();
  togglePlay();
});

async function boot() {
  try {
    const res = await fetch("/api/catalog");
    if (!res.ok) throw new Error("The simulator server did not answer.");
    state.catalog = await res.json();
    if (!state.catalog.controllers.length || !state.catalog.tracks.length) {
      throw new Error("No controllers or maps were found.");
    }
    state.controllerId = state.catalog.controllers[0].id;
    state.trackId = state.catalog.tracks.find((item) => item.id === "hackathon_test")?.id
      || state.catalog.tracks[0].id;
    renderCatalog();
    clearReplay();
    await loadCourse(state.trackId);
  } catch (error) {
    const offline = location.protocol === "file:";
    showError(offline
      ? "Open this page from the simulator: python -m autocross web"
      : error.message);
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (state.playing && state.replay) {
    state.t += dt * state.rate;
    if (state.t >= endTime()) {
      state.t = endTime();
      state.playing = false;
      state.finished = true;
    }
  }
  paint();
  requestAnimationFrame(frame);
}

boot();
requestAnimationFrame(frame);

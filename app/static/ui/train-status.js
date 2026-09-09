/* Watching a training run: the poll loop, the status card and the topbar chip.

   Split from app.js with the form (see training.js). These two halves change for
   different reasons — this one when what a run REPORTS changes, the other when what a
   run is CONFIGURED with does — and the status card is the piece that has to stay
   readable while everything else on the page is being used. */
"use strict";

let pollTimer = null;
let pollInFlight = false;

async function pollTick() {
  // setInterval does not await async ticks; one tick at a time keeps a slow response
  // from overlapping the next.
  if (pollInFlight) return;
  pollInFlight = true;
  try {
    renderTrainStatus(await Api.get("/train/status"));
  } catch { /* transient poll failure: keep the last rendered state */ }
  finally { pollInFlight = false; }
}

// Registered ONCE for the page lifetime (guarded by pollTimer): browsers
// throttle timers in hidden tabs, so refresh immediately on return. Attaching it
// inside startStatusPolling() leaked a listener on every logout->login cycle.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && pollTimer) pollTick();
});

function startStatusPolling() {
  if (pollTimer) return;
  pollTick();
  pollTimer = setInterval(pollTick, 2500);
}
function stopStatusPolling() { clearInterval(pollTimer); pollTimer = null; }

let lastAnnouncedState = "";

/* The job's status is a closed set (see jobs.py), so it can be said in the reader's
   language; `phase` is free-form prose the training pipeline writes and is shown as it
   came. Spelled out rather than built from the value: a key that exists only as a
   template is invisible to the test that proves every string is translated. */
const STATE_KEYS = {
  idle: "trainStatus.state.idle",
  running: "trainStatus.state.running",
  completed: "trainStatus.state.completed",
  error: "trainStatus.state.error",
  stopped: "trainStatus.state.stopped",
};
const stateLabel = (status) => (STATE_KEYS[status] ? t(STATE_KEYS[status]) : status);

/* Announce only STATE TRANSITIONS to screen readers: the status card itself
   re-renders every 2.5s (elapsed/ETA tick up), so making it a live region would
   re-announce the whole card for the entire duration of a training. */
function announceTrainState(s) {
  const key = `${s.status}|${s.phase}|${s.model_name}`;
  if (key === lastAnnouncedState) return;
  lastAnnouncedState = key;
  const el = $("#train-announce");
  if (!el) return;
  const name = s.model_name || "";
  if (s.status === "running") {
    el.textContent = t("trainStatus.announce.running",
                       { name, phase: s.phase || t("trainStatus.phaseStarting") });
  } else if (s.status === "completed") el.textContent = t("trainStatus.announce.completed", { name });
  else if (s.status === "error") {
    el.textContent = t("trainStatus.announce.failed",
                       { message: s.message || t("trainStatus.seeStatus") });
  } else if (s.status === "stopped") el.textContent = t("trainStatus.announce.stopped");
  else el.textContent = "";
}

function renderTrainStatus(s) {
  renderTrainChip(s);
  announceTrainState(s);
  renderQueueLine(s.queued || []);
  const el = $("#train-status");
  const rows = [
    [t("trainStatus.row.status"), stateLabel(s.status)],
    [t("trainStatus.row.phase"), s.phase || "–"],
    [t("trainStatus.row.detail"), s.phase_detail || "–"],
    [t("trainStatus.row.model"), s.model_name || "–"],
    [t("trainStatus.row.elapsed"), s.elapsed_seconds != null ? t("common.seconds", { count: s.elapsed_seconds }) : "–"],
    [t("trainStatus.row.eta"), s.eta_seconds != null ? `~${t("common.seconds", { count: Math.round(s.eta_seconds) })}` : "–"],
  ];
  let html = `
    <div class="progress" role="progressbar" aria-valuenow="${s.progress}" aria-valuemin="0"
         aria-valuemax="100" aria-label="${esc(t("trainStatus.progressLabel"))}"><span data-width="${s.progress}"></span></div>
    <dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
  // elapsed keeps growing even when the thread is dead — only a stale heartbeat
  // (no progress signal from the training thread) reveals a silent stall.
  if (s.status === "running" && s.seconds_since_heartbeat > 120)
    html += `<p class="warn">${t("trainStatus.stalled", { seconds: Math.round(s.seconds_since_heartbeat) })}</p>`;
  if (s.status === "running")
    html += `<button class="small danger" id="train-stop">${t("trainStatus.stop")}</button>`;
  if (s.status === "error") html += `<p class="error">${esc(s.message || t("trainStatus.failed"))}</p>`;
  if (s.status === "completed" && s.results) {
    // An EVALUATION can finish with no metrics at all — when the dataset shares no
    // labels with the model, the server answers `null` rather than a made-up 0.0.
    // Reaching into .f1_macro then threw, and pollTick swallows exceptions, so the card
    // simply froze on the previous state instead of showing the finished run.
    const scores = s.results.metrics || {};
    html += Number.isFinite(scores.f1_macro)
      ? `<p class="ok">${t("trainStatus.doneWithScores", {
           name: esc(s.results.model_name), macro: scores.f1_macro.toFixed(3),
           micro: scores.f1_micro.toFixed(3), labels: s.results.n_labels })}</p>`
      : `<p class="ok">${t("trainStatus.doneNoScores", { name: esc(s.results.model_name) })}</p>`;
  }
  el.innerHTML = html;
  applyBarWidths(el);
  const stop = $("#train-stop");
  if (stop) stop.addEventListener("click", async () => {
    // The server clears the queue as part of stopping; the next poll shows it gone.
    try { await Api.post("/train/stop"); toast(t("trainStatus.stopRequested")); }
    catch (err) { toast(err.message); }
  });
}

/* What is waiting behind the running run — read from the server, so it survives this
   tab being closed, which is the whole reason the queue moved out of the page. */
function renderQueueLine(queued) {
  const el = $("#train-queue");
  el.hidden = !queued.length;
  el.textContent = queued.length
    ? t("trainStatus.queuedOnServer", { names: queued.join(", ") })
    : "";
}


/* Compact status in the topbar so a running training stays visible on EVERY tab. */
function renderTrainChip(s) {
  const chip = $("#train-chip");
  chip.classList.remove("done", "failed");
  if (s.status === "running") {
    const eta = s.eta_seconds != null
      ? ` · ${t("trainStatus.chip.etaLeft", { seconds: Math.round(s.eta_seconds) })}` : "";
    chip.innerHTML = `<span class="mini-bar"><span data-width="${s.progress}"></span></span>
      ${esc(s.model_name || t("trainStatus.chip.fallbackName"))} ${s.progress}%${esc(eta)}`;
    applyBarWidths(chip);
    chip.hidden = false;
  } else if (s.status === "completed" && s.model_name) {
    chip.classList.add("done");
    chip.textContent = t("trainStatus.chip.done", { name: s.model_name });
    chip.hidden = false;
  } else if (s.status === "error") {
    chip.classList.add("failed");
    chip.textContent = t("trainStatus.chip.failed");
    chip.hidden = false;
  } else {
    chip.hidden = true; // idle/stopped: no noise in the topbar
  }
}

"use strict";

const $ = (id) => document.getElementById(id);
const LABELS = { up: "↑ 위", down: "↓ 아래", left: "← 왼쪽", right: "→ 오른쪽" };
const state = { game: null, events: [], phase: "loading", modelBusy: false, running: false,
  working: false, stop: false, browserTimings: new Map() };
const ms = (value) => Number.isFinite(value) ? `${value.toFixed(1)} ms` : "—";
const pct = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";

function showError(message = "") { $("game-error").textContent = message; $("game-error").hidden = !message; }
function message(text) { $("game-message").textContent = text; }
function node(tag, text, className = "") {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  item.className = className;
  return item;
}

async function api(path, method = "GET", body) {
  const response = await fetch(path, { method, cache: "no-store",
    signal: AbortSignal.timeout(45000),
    headers: { "Content-Type": "application/json", "X-Veyra-Playground": "1" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const error = new Error(typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((item) => item.msg).join(" · ") : `요청 실패 (HTTP ${response.status})`);
    error.status = response.status;
    throw error;
  }
  if (!data) throw new Error("서버 응답을 읽지 못했습니다.");
  return data;
}

function controls() {
  const locked = state.running || state.working || Boolean(state.game?.busy);
  const ended = !state.game || state.game.won || state.game.game_over || state.game.limit_reached;
  $("game-auto").disabled = locked || ended || state.phase !== "ready" || state.modelBusy;
  $("game-step").disabled = locked || ended || state.phase !== "ready" || state.modelBusy;
  $("game-auto").textContent = state.running ? "실행 중…" : "자동 실행";
  $("game-stop").hidden = !state.running;
  $("game-stop").disabled = state.stop;
  $("game-stop").textContent = state.stop ? "현재 판단 후 중단" : "중단";
  $("game-new").disabled = locked;
  $("game-settings").disabled = locked;
  $("game-export").disabled = locked || !state.game || !state.events.length;
  document.querySelectorAll("[data-direction]").forEach((button) => { button.disabled = locked || ended; });
}

function renderDecision() {
  const event = [...state.events].reverse().find((item) => item.source === "model");
  const answer = event?.response.answers.move;
  $("decision-direction").textContent = event ? LABELS[event.direction].split(" ")[0] : "—";
  $("decision-outcome").textContent = !event ? "아직 AI를 실행하지 않았습니다."
    : !event.executed ? "모델이 보류하여 방향키를 실행하지 않았습니다."
    : event.moved ? `${LABELS[event.direction]} · 실제 보드가 이동했습니다.` : `${LABELS[event.direction]} · 보드에 변화가 없었습니다.`;
  $("decision-policy").textContent = !event ? "기존 v13 모델 · 게임 추가 학습 없음"
    : event.abstained ? (event.executed ? "보류된 추천 후보를 실험 모드에서 실행했습니다." : "보류 정책 적용됨") : "모델이 반환한 방향을 실행했습니다.";
  $("game-inference").textContent = ms(event?.inference_ms);
  $("game-roundtrip").textContent = ms(event ? state.browserTimings.get(event.revision) : null);
  $("game-median").textContent = ms(state.game?.stats.model_median_ms);
  const probabilities = $("game-probabilities");
  probabilities.replaceChildren();
  Object.entries(LABELS).forEach(([key, label]) => {
    const row = node("div", undefined, `game-probability${event?.direction === key ? " chosen" : ""}`);
    const track = node("div", undefined, "game-probability-track");
    const fill = node("div", undefined, "game-probability-fill");
    fill.style.width = `${Math.max(0, Math.min(100, (answer?.probabilities[key] || 0) * 100))}%`;
    track.append(fill);
    row.append(node("span", label), track, node("span", pct(answer?.probabilities[key])));
    probabilities.append(row);
  });
  $("decision-frame").hidden = !event;
  if (event) {
    $("decision-frame").src = event.frame_url;
    $("game-request").textContent = JSON.stringify({ image_sha256: event.image_sha256, request: event.request }, null, 2);
  } else { $("decision-frame").removeAttribute("src"); $("game-request").textContent = "실행 후 표시됩니다."; }
}

function renderTrace() {
  const list = $("game-trace");
  list.replaceChildren();
  for (const event of state.events.slice(-100).reverse()) {
    const row = node("tr");
    if (event.abstained) row.className = "held";
    const answer = event.response?.answers.move;
    const outcome = !event.executed ? "보류 · 미실행" : event.moved ? `이동 · +${event.score_gain}` : "변화 없음";
    [String(event.revision + 1), event.source === "model" ? `Veyra${event.abstained ? " · 보류" : ""}` : "직접 조작",
      LABELS[event.direction], outcome, pct(answer?.probabilities[event.direction]), ms(event.inference_ms), String(event.score_after)]
      .forEach((value) => row.append(node("td", value)));
    list.append(row);
  }
  if (!state.events.length) {
    const row = node("tr"); const cell = node("td", "AI 한 수 또는 자동 실행을 눌러 시작하세요.");
    cell.colSpan = 7; row.append(cell); list.append(row);
  }
}

async function renderGame(game) {
  state.game = game;
  const last = game.last_event;
  if (last && !state.events.some((event) => event.revision === last.revision)) state.events.push(last);
  $("game-score").textContent = game.score.toLocaleString("ko-KR");
  $("game-tile").textContent = game.max_tile;
  $("game-moves").textContent = `${game.stats.valid_moves} / ${game.stats.attempts}`;
  $("game-board").src = game.frame_url;
  $("game-board").alt = `현재 2048 보드. ${game.board.map((row, i) => `${i + 1}행: ${row.map((value) => value || "빈 칸").join(", ")}`).join(". ")}`;
  renderDecision(); renderTrace(); controls();
  try { await $("game-board").decode(); } catch { /* Next action reports an expired session. */ }
}

function terminalMessage(game) {
  if (game.won) return "2048 달성! 게임 엔진에서 목표 타일을 확인했습니다.";
  if (game.game_over) return "게임 종료 · 더 이상 이동할 수 없습니다.";
  if (game.limit_reached) return `세션의 ${game.event_limit.toLocaleString("ko-KR")}단계 한도에 도달했습니다. 기록을 저장하고 새 게임을 시작하세요.`;
  if (game.last_event?.source === "model" && !game.last_event.executed) return "모델이 판단을 보류했습니다. 직접 조작하거나 설정을 바꾸어 계속할 수 있습니다.";
  if (game.stats.no_progress_streak >= 4) return "4회 연속 보드에 변화가 없어 자동 실행을 멈췄습니다. 직접 한 수를 두거나 새 게임을 시작하세요.";
  return null;
}

async function newGame() {
  if (state.running || state.working) return;
  const seed = Number($("game-seed").value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) { showError("시드는 0~4294967295의 정수로 입력하세요."); return; }
  state.working = true; controls(); showError();
  try {
    const previous = state.game;
    if (previous) {
      try { await api(`/api/2048/games/${previous.id}`, "DELETE"); }
      catch (error) { if (error.status !== 404) throw error; }
      state.game = null;
    }
    const game = await api("/api/2048/games", "POST", { seed, image_size: Number($("game-size").value), theme: $("game-theme").value });
    state.events = []; state.browserTimings.clear();
    history.replaceState(null, "", `/2048?game=${encodeURIComponent(game.id)}`);
    await renderGame(game);
    message("준비되었습니다. AI 한 수로 살펴보거나 자동 실행을 시작하세요.");
  } catch (error) { showError(error.message); }
  finally { state.working = false; controls(); }
}

async function stepModel() {
  const start = performance.now();
  const revision = state.game.revision;
  const next = await api(`/api/2048/games/${state.game.id}/step`, "POST", {
    revision, respect_abstention: $("game-abstention").checked });
  state.modelBusy = false;
  state.browserTimings.set(revision, performance.now() - start);
  await renderGame(next);
  return next;
}

async function recoverState() {
  if (!state.game) return;
  try {
    const report = await api(`/api/2048/games/${state.game.id}/report`);
    state.events = report.events; await renderGame(report.final);
  } catch (error) {
    // Keep the original action error; never retry an uncertain action automatically.
    if (error.status === 404) {
      state.game = null;
      message("게임 세션이 만료되었습니다. 새 게임을 눌러 시작하세요.");
      controls();
    }
  }
}

async function run(automatic) {
  if (!state.game || state.running || state.working || state.phase !== "ready") return;
  state.running = automatic; state.working = true; state.stop = false; controls(); showError();
  const budget = automatic ? Number($("game-budget").value) : 1;
  let completed = 0;
  try {
    while (completed < budget && !state.stop) {
      message(`${automatic ? "자동 실행" : "AI 판단"} · ${completed + 1} / ${budget} · 이미지를 보고 방향을 고르는 중`);
      const game = await stepModel();
      completed += 1;
      const ended = terminalMessage(game);
      if (ended) { message(ended); return; }
    }
    message(state.stop ? `${completed}회 판단 후 중단했습니다.` : `${completed}회 판단 완료 · 점수 ${state.game.score}, 가장 큰 타일 ${state.game.max_tile}`);
  } catch (error) {
    await recoverState(); showError(error.message);
    message("실행을 중단했습니다. 요청은 자동 재실행하지 않습니다. 서버 연결과 현재 보드를 확인하세요.");
  } finally { state.running = false; state.working = false; state.stop = false; controls(); }
}

async function manual(direction) {
  if (!state.game || state.running || state.working || state.game.won || state.game.game_over || state.game.limit_reached) return;
  state.working = true; controls(); showError();
  try {
    const game = await api(`/api/2048/games/${state.game.id}/move`, "POST", { revision: state.game.revision, direction });
    await renderGame(game);
    message(terminalMessage(game) || `${LABELS[direction]} · ${game.last_event.moved ? "직접 이동했습니다." : "움직일 수 없는 방향입니다."}`);
  } catch (error) { await recoverState(); showError(error.message); }
  finally { state.working = false; controls(); }
}

async function exportReport() {
  if (!state.game || state.running || state.working) return;
  try {
    const report = await api(`/api/2048/games/${state.game.id}/report`);
    report.browser_roundtrip_ms = Object.fromEntries(state.browserTimings);
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url;
    link.download = `veyra-2048-${report.final.config.seed}-${report.final.id.slice(0, 8)}.json`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { showError(error.message); }
}

async function poll() {
  try {
    const status = await api("/api/status");
    state.phase = status.phase; state.modelBusy = status.busy;
    $("connection").className = `connection ${status.phase}`;
    $("connection-label").textContent = status.phase === "ready" ? (status.busy ? "모델 사용 중" : "모델 연결됨") : status.phase === "loading" ? "모델 준비 중" : "모델 준비 실패";
    $("device-label").textContent = `${status.checkpoint.replace("veyra-backbone-recovery-", "")} · ${status.device || "GPU"}`;
    if (state.game?.busy && !state.running && !state.working) await recoverState();
  } catch { state.phase = "offline"; $("connection").className = "connection error"; $("connection-label").textContent = "연결 끊김"; }
  finally { controls(); setTimeout(poll, 3000); }
}

$("game-auto").addEventListener("click", () => run(true));
$("game-step").addEventListener("click", () => run(false));
$("game-stop").addEventListener("click", () => { state.stop = true; controls(); });
$("game-new").addEventListener("click", newGame);
$("game-export").addEventListener("click", exportReport);
document.querySelectorAll("[data-direction]").forEach((button) => button.addEventListener("click", () => manual(button.dataset.direction)));
document.addEventListener("keydown", (event) => {
  const direction = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" }[event.key];
  if (!direction || /INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
  event.preventDefault(); manual(direction);
});

async function initialize() {
  const id = new URLSearchParams(location.search).get("game");
  if (!id || !/^[a-f0-9]{32}$/.test(id)) { await newGame(); return; }
  state.working = true; controls();
  try {
    const report = await api(`/api/2048/games/${id}/report`);
    state.events = report.events;
    $("game-seed").value = report.final.config.seed;
    $("game-size").value = report.final.config.image_size;
    $("game-theme").value = report.final.config.theme;
    await renderGame(report.final);
    message(terminalMessage(report.final) || "저장된 서버 세션을 불러왔습니다. 이어서 실행할 수 있습니다.");
  } catch (error) { showError(error.message); message("새 게임을 눌러 시작하세요."); }
  finally { state.working = false; controls(); }
}
renderDecision(); poll(); initialize();

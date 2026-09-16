import { Wllama, LoggerWithoutDebug } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/index.js";

// wasm-from-cdn.js is inlined in the bundle; build the config manually:
const CONFIG_PATHS = {
  default: "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/wasm/wllama.wasm",
};

// ================== TOGGLES ==================
// Speculative decoding: DOES NOT WORK on this WASM build — the fixed 4-worker
// pthread pool deadlocks when the draft context requests compute threads.
// Confirmed via log; do not enable.
const ENABLE_SPEC_DECODING = true;

// ================== MODELS ==================
const MODELS = {
  "gemma-270m": {
    label: "Gemma 3 270M — fast",
    url: "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_0.gguf",
    n_ctx: 2048,
    threads: 4,
    cache_k: "q8_0",
    cache_v: "q8_0",
    draft: null,
  },
  "qwen-0.5b": {
    label: "Qwen2.5 0.5B — balanced",
    url: "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf",
    n_ctx: 4096,
    threads: 4,
    cache_k: "q8_0",
    cache_v: "q8_0",
    draft: null,
  },
  "gemma-1b": {
    label: "Gemma 3 1B — smart",
    url: "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf",
    n_ctx: 4096,
    threads: 4,
    cache_k: "q4_0",
    cache_v: "q4_0",
    draft: "gemma-270m", // unused while ENABLE_SPEC_DECODING is false
  },
};

const MODEL_CHOICE_KEY = "wllama-model-choice";
let currentModelKey = localStorage.getItem(MODEL_CHOICE_KEY) || "gemma-1b";
if (!MODELS[currentModelKey]) currentModelKey = "gemma-1b";

// ================== CONFIG ==================
let N_PREDICT = 512;

const COST_TOKEN_PER_1M = 0.20;
const COST_CPU_HOUR = 0.05;
const COST_KEY = "gemma-270m-wllama-cost-v1";
const CHAT_KEY = "gemma-270m-wllama-chat-v1";

// ================== TYPEWRITER ==================
// Typing speed tracks the model's LIVE output rate (measured as tokens
// arrive), so fast models type fast and slow models type at reading pace —
// but hard caps guarantee it's always visibly character-by-character.
// Tune these:
const TYPE_BASE_CPS = 40;    // floor: minimum chars/sec (reading speed)
const TYPE_MAX_CPS = 150;    // cap while generating (≈2 chars/frame at 60fps)
const TYPE_DRAIN_MAX = 200;  // cap when draining after generation ends

let typeEl = null;
let typeQueue = "";
let typeFinished = false;
let typeRaf = null;
let typeLastTs = 0;
let typeSpeed = TYPE_BASE_CPS;
let typeAccum = 0;
let drainResolve = null;
let inRate = 0;      // smoothed live incoming chars/sec from the model
let lastInTs = 0;

function startTyping(el) {
  finishTyping();
  typeEl = el;
  typeQueue = "";
  typeFinished = false;
  typeSpeed = TYPE_BASE_CPS;
  typeAccum = 0;
  inRate = 0;
  lastInTs = 0;
  el.classList.add("typing");
  typeLastTs = performance.now();
  typeRaf = requestAnimationFrame(typeFrame);
}

// call from onData — keeps a smoothed estimate of how fast the model writes
function feedTypingRate(piece) {
  const now = performance.now();
  if (lastInTs && piece) {
    const dt = (now - lastInTs) / 1000;
    if (dt > 0 && dt < 1) {
      const inst = piece.length / dt;
      inRate = inRate ? inRate * 0.75 + inst * 0.25 : inst;
    }
  }
  lastInTs = now;
}

function typeFrame(ts) {
  if (!typeEl) return;
  const dt = Math.min((ts - typeLastTs) / 1000, 0.1);
  typeLastTs = ts;

  // don't bank speed while there's nothing to type
  if (!typeQueue.length && !typeFinished) {
    typeAccum = 0;
  }

  let target;
  if (typeFinished) {
    // generation done: drain the rest briskly but visibly
    // (a full essay drains in ~5–10s, never instantly)
    target = Math.min(TYPE_DRAIN_MAX, Math.max(90, typeQueue.length / 5));
  } else {
    // pace with the model's live rate + gentle catch-up on backlog,
    // hard-capped so it always looks like typing
    target = Math.min(
      TYPE_MAX_CPS,
      TYPE_BASE_CPS + inRate * 0.9 + typeQueue.length * 1.5
    );
  }

  // smooth acceleration toward the target (no jumps)
  typeSpeed += (target - typeSpeed) * Math.min(1, dt * 6);

  typeAccum += typeSpeed * dt;
  const n = Math.floor(typeAccum);
  if (n > 0 && typeQueue.length) {
    const take = Math.min(n, typeQueue.length);
    typeEl.textContent += typeQueue.slice(0, take);
    typeQueue = typeQueue.slice(take);
    typeAccum -= take;
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  if (typeFinished && !typeQueue.length) { finishTyping(); return; }
  typeRaf = requestAnimationFrame(typeFrame);
}

function finishTyping() {
  if (typeRaf) { cancelAnimationFrame(typeRaf); typeRaf = null; }
  if (typeEl) { typeEl.classList.remove("typing"); typeEl = null; }
  if (drainResolve) { drainResolve(); drainResolve = null; }
}

function typeFinish() {
  typeFinished = true;
  if (!typeRaf) return Promise.resolve();
  return new Promise((resolve) => { drainResolve = resolve; });
}

function abortTyping() {
  typeQueue = "";
  typeFinished = true;
  finishTyping();
}

// ================== ON-PAGE LOG ==================
const LOG_KEY = "wllama-log-v1";
let logEntries = [];
try { logEntries = JSON.parse(localStorage.getItem(LOG_KEY) || "[]"); } catch {}
if (!Array.isArray(logEntries)) logEntries = [];

let logPanelEl = null, logListEl = null, logSaveTimer = null;

function fmtLogTime(t) {
  return new Date(t).toTimeString().slice(0, 8);
}

function logLine(tag, ...args) {
  const msg = args.map((a) => {
    if (a instanceof Error) {
      return a.message + (a.stack ? " | " + a.stack.split("\n").slice(1, 3).join(" | ") : "");
    }
    if (typeof a === "object") { try { return JSON.stringify(a); } catch { return String(a); } }
    return String(a);
  }).join(" ");
  const entry = { t: Date.now(), tag, msg };
  logEntries.push(entry);
  if (logEntries.length > 200) logEntries = logEntries.slice(-200);
  if (logListEl) appendLogDom(entry);
  clearTimeout(logSaveTimer);
  logSaveTimer = setTimeout(() => {
    try { localStorage.setItem(LOG_KEY, JSON.stringify(logEntries.slice(-200))); } catch {}
  }, 500);
}

function appendLogDom(entry) {
  const div = document.createElement("div");
  div.textContent = `[${fmtLogTime(entry.t)}] ${entry.tag}: ${entry.msg}`;
  logListEl.appendChild(div);
  while (logListEl.children.length > 200) logListEl.firstChild.remove();
  logListEl.scrollTop = logListEl.scrollHeight;
}

function buildLogUI() {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "Logs";
  btn.title = "Show on-page error log";
  btn.style.cssText =
    "background:#333a47;color:#e6e6e6;border:none;border-radius:6px;" +
    "padding:1px 8px;font:inherit;font-size:12px;cursor:pointer;";
  statusEl.parentNode.appendChild(btn);

  logPanelEl = document.createElement("div");
  logPanelEl.style.cssText =
    "display:none;position:fixed;left:0;right:0;bottom:0;height:45vh;" +
    "background:rgba(8,10,14,0.97);border-top:1px solid #333a47;" +
    "z-index:9999;padding:8px 10px;box-sizing:border-box;" +
    "flex-direction:column;gap:6px;font-family:monospace;";
  const head = document.createElement("div");
  head.style.cssText = "display:flex;gap:8px;align-items:center;font-size:12px;color:#9aa3b2;";
  const title = document.createElement("span");
  title.textContent = "Log (persists across reloads)";
  const clear = document.createElement("button");
  clear.type = "button";
  clear.textContent = "Clear";
  clear.style.cssText = "background:#333a47;color:#e6e6e6;border:none;border-radius:6px;padding:1px 8px;font:inherit;font-size:11px;cursor:pointer;";
  clear.addEventListener("click", () => {
    logEntries = [];
    try { localStorage.removeItem(LOG_KEY); } catch {}
    logListEl.replaceChildren();
  });
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.style.cssText = clear.style.cssText;
  close.addEventListener("click", () => { logPanelEl.style.display = "none"; });
  head.append(title, clear, close);

  logListEl = document.createElement("div");
  logListEl.style.cssText =
    "flex:1;overflow-y:auto;font-size:11px;line-height:1.5;color:#c9d1e0;" +
    "white-space:pre-wrap;word-break:break-all;";
  for (const e of logEntries) appendLogDom(e);

  logPanelEl.append(head, logListEl);
  document.body.appendChild(logPanelEl);

  btn.addEventListener("click", () => {
    const open = logPanelEl.style.display === "flex";
    logPanelEl.style.display = open ? "none" : "flex";
    if (!open) logListEl.scrollTop = logListEl.scrollHeight;
  });
}

window.addEventListener("error", (e) => {
  logLine("window", e.message || "error", `@${e.filename}:${e.lineno}`);
});
window.addEventListener("unhandledrejection", (e) => {
  const r = e.reason;
  logLine("promise", r instanceof Error ? r.message : String(r));
});

function makeLogger() {
  const wrap = (level) => (...args) => {
    logLine("wllama", `[${level}]`, ...args);
    try { console[level]?.(...args); } catch {}
  };
  return { debug: wrap("debug"), log: wrap("log"), warn: wrap("warn"), error: wrap("error") };
}

// ================== ELEMENTS / STATE ==================
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), loadbarEl = $("loadbar"), loadfillEl = $("loadfill");
const backendEl = $("backend"), ramEl = $("ram"), storageEl = $("storage");
const speedEl = $("speed"), costEl = $("cost"), chatEl = $("chat");
const formEl = $("form"), inputEl = $("input"), sendEl = $("send"), clearEl = $("clear");

let wllama = null;
let conversation = [];
let generating = false;
let modelLoading = false;
let lastPrompt = "";
let totalCost = Number(localStorage.getItem(COST_KEY) || "0");

let sharedCacheManager = null;
let sharedModelManager = null;

function getWllama() {
  const cfg = { logger: makeLogger() };
  if (!sharedCacheManager) {
    const first = new Wllama(CONFIG_PATHS, cfg);
    sharedCacheManager = first.cacheManager;
    sharedModelManager = first.modelManager;
    return first;
  }
  return new Wllama(CONFIG_PATHS, {
    ...cfg,
    cacheManager: sharedCacheManager,
    modelManager: sharedModelManager,
  });
}

// ================== MODEL SELECTOR ==================
const modelSelect = document.createElement("select");
modelSelect.id = "modelSelect";
modelSelect.title = "Switch model";
modelSelect.style.cssText =
  "background:#111318;color:#e6e6e6;border:1px solid #333a47;" +
  "border-radius:6px;padding:2px 6px;font:inherit;font-size:12px;";
for (const [key, m] of Object.entries(MODELS)) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = m.label;
  modelSelect.appendChild(opt);
}
modelSelect.value = currentModelKey;
statusEl.parentNode.appendChild(modelSelect);

modelSelect.addEventListener("change", () => {
  const key = modelSelect.value;
  if (key === currentModelKey) return;
  if (generating || modelLoading) {
    modelSelect.value = currentModelKey;
    return;
  }
  switchModel(key);
});

function updateModelHeading() {
  const label = MODELS[currentModelKey].label.split(" — ")[0];
  const h1 = document.querySelector("h1");
  if (h1) h1.textContent = `${label} — Local Browser AI`;
  document.title = `${label} — Local`;
}

async function switchModel(key) {
  const prevKey = currentModelKey;
  currentModelKey = key;
  localStorage.setItem(MODEL_CHOICE_KEY, key);
  updateModelHeading();
  logLine("app", "Switching model to", key);

  document.querySelectorAll(".continue-notice").forEach((n) => n.remove());
  N_PREDICT = 512;

  try {
    setStatus("Switching model…");
    setProgress(0);
    try { await wllama?.exit?.(); } catch (e) { logLine("app", "exit() during switch failed:", e?.message || e); }
    wllama = null;
    await loadCurrentModel();
  } catch (error) {
    logLine("app", "Model switch FAILED:", error?.message || error);
    currentModelKey = prevKey;
    localStorage.setItem(MODEL_CHOICE_KEY, prevKey);
    modelSelect.value = prevKey;
    updateModelHeading();
    try {
      setStatus("Switch failed — reloading previous model…");
      try { await wllama?.exit?.(); } catch {}
      wllama = null;
      await loadCurrentModel();
    } catch (e2) {
      logLine("app", "Revert ALSO failed:", e2?.message || e2);
      setStatus(`Switch failed: ${error?.message || error}`);
    }
  }
}

// ================== UI ==================
function setStatus(text) { statusEl.textContent = text; }
function setProgress(percent) {
  loadbarEl.style.display = "block";
  loadfillEl.style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
}
function hideProgress() { loadbarEl.style.display = "none"; }
function addMessage(role, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

// ================== MEMORY / STORAGE / COST ==================
function updateMemory() {
  if (performance.memory) {
    const used = performance.memory.usedJSHeapSize / 1024 / 1024;
    const limit = performance.memory.jsHeapSizeLimit / 1024 / 1024;
    ramEl.textContent = `${used.toFixed(0)} / ${limit.toFixed(0)} MB`;
  } else ramEl.textContent = "not exposed";
}
async function updateStorage() {
  try {
    if (!navigator.storage?.estimate) { storageEl.textContent = "unavailable"; return; }
    const info = await navigator.storage.estimate();
    const used = (info.usage || 0) / 1024 / 1024;
    const quota = (info.quota || 0) / 1024 / 1024;
    storageEl.textContent = quota > 0 ? `${used.toFixed(0)} / ${quota.toFixed(0)} MB` : `${used.toFixed(0)} MB`;
  } catch { storageEl.textContent = "unavailable"; }
}
function updateCost() { costEl.textContent = `$${totalCost.toFixed(4)}`; }

// ================== CHAT STORAGE ==================
function saveConversation() {
  try { localStorage.setItem(CHAT_KEY, JSON.stringify(conversation.slice(-24))); }
  catch (e) { console.warn("Could not save chat:", e); }
}
function restoreConversation() {
  try {
    const raw = localStorage.getItem(CHAT_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved)) return;
    conversation = saved;
    for (const m of conversation) addMessage(m.role, m.content);
  } catch (e) { console.warn("Could not restore chat:", e); }
}

// ================== MODEL LOADING ==================
async function loadCurrentModel() {
  const model = MODELS[currentModelKey];
  modelLoading = true;
  modelSelect.disabled = true;
  inputEl.disabled = true;
  sendEl.disabled = true;

  try {
    const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
    logLine("app", `Loading model: ${currentModelKey}`, `ctx=${model.n_ctx}`, `threads=${model.threads}`);

    setStatus(`Loading ${model.label}…`);
    setProgress(0);
    wllama = getWllama();

    await wllama.loadModelFromUrl(model.url, {
      n_ctx: model.n_ctx,
      n_gpu_layers: 0,       // CPU/WASM only — skips WebGPU init entirely
      n_batch: 2048,          // prefill batching
      n_ubatch: 512,
	  n_threads: 2,
	  spec_draft_model: "models/model-00002-of-00002.gguf",
	  spec_draft_threads: 1,
	  spec_draft_threads_batch: 1,
      flash_attn: true,       // faster attention; required for quantized KV cache
      cache_type_k: model.cache_k,
      cache_type_v: model.cache_v,
      ctx_shift: true,
      progressCallback: ({ loaded, total }) => {
        setProgress((loaded / total) * 100);
        setStatus(`Downloading ${model.label}… ${Math.round((loaded / total) * 100)}%`);
      },
    });

    hideProgress();
    backendEl.textContent = isolated
      ? `wllama CPU • ${model.threads} threads`
      : "wllama CPU • 1 thread (isolation FAILED)";
    setStatus("Ready");
    logLine("app", "Model ready:", model.label);
    inputEl.disabled = false;
    sendEl.disabled = false;
    inputEl.focus();
    updateMemory();
    await updateStorage();
    updateCost();
  } catch (error) {
    logLine("app", "Model load FAILED:", error?.message || error);
    setStatus(`Startup error: ${error?.message || error}`);
    throw error;
  } finally {
    modelLoading = false;
    modelSelect.disabled = false;
  }
}

// ================== GENERATION ==================
async function generateResponse(prompt, replyElement) {
  const messages = [
    { role: "system", content: "You are a helpful local assistant running in the browser." },
    ...conversation.slice(-6),
    { role: "user", content: prompt },
  ];

  const started = performance.now();
  let firstTokenAt = null, lastTokenAt = null, generatedText = "";
  let finishReason = null;

  startTyping(replyElement);
  logLine("app", "Generation start:", currentModelKey, `max_tokens=${N_PREDICT}`);

  const result = await wllama.createChatCompletion({
    messages,
    max_tokens: N_PREDICT,
    temperature: 1.0,
    top_k: 64,
    top_p: 0.95,
    stream: true,
    onData: (chunk) => {
      const now = performance.now();
      if (firstTokenAt === null) firstTokenAt = now;
      lastTokenAt = now;
      const piece = chunk.choices?.[0]?.delta?.content ?? "";
      generatedText += piece;
      typeQueue += piece;
      feedTypingRate(piece); // typewriter paces itself to the live model speed
      if (chunk.choices?.[0]?.finish_reason) {
        finishReason = chunk.choices[0].finish_reason;
      }
    },
  });

  if (!finishReason && result?.choices?.[0]?.finish_reason) {
    finishReason = result.choices[0].finish_reason;
  }

  const completionTokens = result?.usage?.completion_tokens;
  const truncated = finishReason === "length" ||
    (Number.isFinite(completionTokens) && completionTokens >= N_PREDICT);

  await typeFinish(); // let the remaining letters finish appearing

  const ended = performance.now();
  const firstAt = firstTokenAt ?? ended, lastAt = lastTokenAt ?? ended;
  const prefillSeconds = Math.max((firstAt - started) / 1000, 0.001);
  const genSeconds = Math.max((lastAt - firstAt) / 1000, 0.001);

  const outputTokens = (Number.isFinite(completionTokens) && completionTokens > 0)
    ? completionTokens
    : Math.max(1, Math.round(generatedText.length / 4));
  const tokensPerSecond = outputTokens / genSeconds;

  const truncNote = truncated ? " • TRUNCATED" : "";
  speedEl.textContent =
    `${outputTokens} tok • ${tokensPerSecond.toFixed(1)} tok/s • prefill ${prefillSeconds.toFixed(1)}s${truncNote}`;

  logLine("app", `Generation done: ${outputTokens} tok, ${tokensPerSecond.toFixed(1)} tok/s, prefill ${prefillSeconds.toFixed(1)}s, finish=${finishReason || "unknown"}`);

  const requestCost =
    (outputTokens / 1_000_000) * COST_TOKEN_PER_1M +
    ((prefillSeconds + genSeconds) / 3600) * COST_CPU_HOUR;
  totalCost += requestCost;
  localStorage.setItem(COST_KEY, String(totalCost));
  updateCost();

  conversation.push(
    { role: "user", content: prompt },
    { role: "assistant", content: generatedText },
  );
  saveConversation();

  if (truncated) {
    const nextLimit = Math.min(N_PREDICT * 2, 2048);
    const notice = document.createElement("div");
    notice.className = "continue-notice";
    notice.style.cssText =
      "align-self:flex-start;display:flex;gap:10px;align-items:center;" +
      "font-size:12px;color:#f0b45f;margin-top:-4px;";
    const span = document.createElement("span");
    span.textContent = `Cut off at ${N_PREDICT} tokens`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = `Continue with limit ${nextLimit}`;
    btn.style.cssText =
      "background:#333a47;color:#e6e6e6;border:none;border-radius:8px;" +
      "padding:4px 10px;font:inherit;font-size:12px;cursor:pointer;";
    btn.addEventListener("click", () => continueLast(nextLimit, replyElement, notice));
    notice.append(span, btn);
    chatEl.appendChild(notice);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  await updateStorage();
  updateMemory();
}

async function continueLast(newLimit, replyElement, noticeEl) {
  if (generating || !wllama) return;
  generating = true;
  modelSelect.disabled = true;
  noticeEl?.remove();
  N_PREDICT = newLimit;
  inputEl.disabled = true;
  sendEl.disabled = true;

  if (conversation.at(-1)?.role === "assistant") conversation.pop();
  if (conversation.at(-1)?.role === "user" && conversation.at(-1).content === lastPrompt) {
    conversation.pop();
  }
  replyElement.textContent = "";

  try {
    setStatus("Regenerating with higher limit…");
    await generateResponse(lastPrompt, replyElement);
    setStatus("Ready");
  } catch (error) {
    logLine("app", "Continue failed:", error?.message || error);
    abortTyping();
    replyElement.textContent = `Error: ${error?.message || error}`;
    setStatus("Continue failed");
  } finally {
    generating = false;
    modelSelect.disabled = modelLoading;
    inputEl.disabled = false;
    sendEl.disabled = false;
    inputEl.focus();
  }
}

// ================== SEND ==================
async function sendMessage() {
  if (generating) return;
  const prompt = inputEl.value.trim();
  if (!prompt) return;
  if (!wllama) { setStatus("Model is still loading…"); return; }
  generating = true;
  modelSelect.disabled = true;
  lastPrompt = prompt;
  inputEl.value = "";
  inputEl.disabled = true;
  sendEl.disabled = true;
  addMessage("user", prompt);
  const replyElement = addMessage("assistant", "");
  try {
    setStatus("Generating…");
    await generateResponse(prompt, replyElement);
    setStatus("Ready");
  } catch (error) {
    logLine("app", "Generation failed:", error?.message || error);
    abortTyping();
    replyElement.textContent = `Error: ${error?.message || error}`;
    setStatus("Generation failed");
  } finally {
    generating = false;
    modelSelect.disabled = modelLoading;
    inputEl.disabled = false;
    sendEl.disabled = false;
    inputEl.focus();
  }
}

// ================== EVENTS ==================
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
formEl.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
clearEl.addEventListener("click", () => {
  abortTyping();
  conversation = [];
  localStorage.removeItem(CHAT_KEY);
  chatEl.replaceChildren();
});
setInterval(() => { updateMemory(); updateStorage(); }, 1500);

// ================== STARTUP ==================
(async () => {
  buildLogUI();
  logLine("app", "=== App started ===");
  try {
    restoreConversation();
    updateModelHeading();
    updateCost();
    updateMemory();
    await updateStorage();
    await loadCurrentModel();
  } catch (error) {
    logLine("app", "STARTUP ERROR:", error?.message || error);
    setStatus(`Startup error: ${error?.message || error}`);
  }
})();
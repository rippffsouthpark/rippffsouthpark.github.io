import { Wllama, LoggerWithoutDebug } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/index.js";

// wasm-from-cdn.js is inlined in the bundle; build the config manually:
const CONFIG_PATHS = {
  default: "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/wasm/wllama.wasm",
};

// ================== MODELS ==================
const MODELS = {
  "gemma-270m": {
    label: "Gemma 3 270M — fast",
    url: "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_0.gguf",
  },
  "qwen-0.5b": {
    label: "Qwen2.5 0.5B — balanced",
    url: "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf",
  },
  "gemma-1b": {
    label: "Gemma 3 1B — smart",
    url: "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf",
  },
};
const MODEL_CHOICE_KEY = "wllama-model-choice";
let currentModelKey = localStorage.getItem(MODEL_CHOICE_KEY) || "gemma-1b";
if (!MODELS[currentModelKey]) currentModelKey = "gemma-1b";

// ================== CONFIG ==================
let N_PREDICT = 512;       // max tokens per reply — can be raised at runtime by "Continue"
const THREADS = 4;        // tune: try 2, 4, 6
const N_CTX = 4096;

const TYPE_BASE_CPS = 28;  // baseline typing speed, chars/sec

const COST_TOKEN_PER_1M = 0.20;
const COST_CPU_HOUR = 0.05;
const COST_KEY = "gemma-270m-wllama-cost-v1";
const CHAT_KEY = "gemma-270m-wllama-chat-v1";

// ================== TYPEWRITER (smooth, rAF-based) ==================
let typeEl = null;
let typeQueue = "";
let typeFinished = false;
let typeRaf = null;
let typeLastTs = 0;
let typeSpeed = TYPE_BASE_CPS;   // chars/sec, smoothly interpolated
let typeAccum = 0;
let drainResolve = null;

function startTyping(el) {
  finishTyping();
  typeEl = el;
  typeQueue = "";
  typeFinished = false;
  typeSpeed = TYPE_BASE_CPS;
  typeAccum = 0;
  el.classList.add("typing");
  typeLastTs = performance.now();
  typeRaf = requestAnimationFrame(typeFrame);
}

function typeFrame(ts) {
  if (!typeEl) return;
  const dt = Math.min((ts - typeLastTs) / 1000, 0.1); // clamp tab-switch jumps
  typeLastTs = ts;

  // target speed: readable when idle-ish, speeds up with backlog,
  // drains fast-but-visibly once generation has ended
  let target;
  if (typeFinished) target = Math.max(80, typeQueue.length * 2);
  else if (typeQueue.length > 80) target = typeQueue.length * 0.9; // track the generator
  else if (typeQueue.length > 30) target = 55;
  else target = TYPE_BASE_CPS;

  // smooth acceleration/deceleration instead of discrete jumps
  typeSpeed += (target - typeSpeed) * Math.min(1, dt * 4);

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

// resolves when generation is done AND all letters have been displayed
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

// ================== ELEMENTS / STATE ==================
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), loadbarEl = $("loadbar"), loadfillEl = $("loadfill");
const backendEl = $("backend"), ramEl = $("ram"), storageEl = $("storage");
const speedEl = $("speed"), costEl = $("cost"), chatEl = $("chat");
const formEl = $("form"), inputEl = $("input"), sendEl = $("send"), clearEl = $("clear");

let wllama = null;
let conversation = [];
let generating = false;
let lastPrompt = "";
let totalCost = Number(localStorage.getItem(COST_KEY) || "0");

// shared cache/model managers so switching models doesn't re-download
let sharedCacheManager = null;
let sharedModelManager = null;

function getWllama() {
  if (!sharedCacheManager) {
    const first = new Wllama(CONFIG_PATHS, { logger: LoggerWithoutDebug });
    sharedCacheManager = first.cacheManager;
    sharedModelManager = first.modelManager;
    return first;
  }
  return new Wllama(CONFIG_PATHS, {
    logger: LoggerWithoutDebug,
    cacheManager: sharedCacheManager,
    modelManager: sharedModelManager,
  });
}

// ================== MODEL SELECTOR (built dynamically, no HTML edit needed) ==================
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
  if (generating || key === currentModelKey) {
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
  currentModelKey = key;
  localStorage.setItem(MODEL_CHOICE_KEY, key);
  updateModelHeading();
  inputEl.disabled = true;
  sendEl.disabled = true;
  try {
    setStatus("Switching model…");
    setProgress(0);
    try { await wllama?.exit?.(); } catch (e) { console.warn("exit failed:", e); }
    wllama = null;
    await loadCurrentModel();
  } catch (error) {
    console.error("Model switch failed:", error);
    setStatus(`Switch failed: ${error?.message || error}`);
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
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  backendEl.textContent = isolated
    ? `wllama CPU • ${THREADS} threads`
    : "wllama CPU • 1 thread (isolation FAILED)";

  setStatus(`Loading ${model.label}…`);
  setProgress(0);

  wllama = getWllama();

  await wllama.loadModelFromUrl(model.url, {
    n_ctx: N_CTX,
    n_threads: THREADS,
    n_gpu_layers: 0,        // CPU/WASM only — skips WebGPU init entirely
    n_batch: 2048,          // prefill batching
    n_ubatch: 512,
    flash_attn: true,       // faster attention; required for quantized KV cache
    cache_type_k: "q8_0",   // halves KV memory
    cache_type_v: "q8_0",
    ctx_shift: true,        // drop oldest tokens when context fills instead of erroring
    progressCallback: ({ loaded, total }) => {
      setProgress((loaded / total) * 100);
      setStatus(`Downloading ${model.label}… ${Math.round((loaded / total) * 100)}%`);
    },
  });

  hideProgress();
  setStatus("Ready");
  inputEl.disabled = false;
  sendEl.disabled = false;
  inputEl.focus();
  updateMemory();
  await updateStorage();
  updateCost();
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

  // ---- "response cut off" continuation offer ----
  if (truncated) {
    const nextLimit = Math.min(N_PREDICT * 2, 2048);
    const notice = document.createElement("div");
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

// Re-generates the last exchange with a higher max_tokens.
// NOTE: this re-pays prefill (the essay gets reprocessed) — that's the
// cost of the simple approach; it always works and keeps the chat clean.
async function continueLast(newLimit, replyElement, noticeEl) {
  if (generating || !wllama) return;
  generating = true;
  noticeEl?.remove();
  N_PREDICT = newLimit;
  inputEl.disabled = true;
  sendEl.disabled = true;

  // drop the truncated assistant reply + the user turn that produced it,
  // so generateResponse re-adds them cleanly
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
    console.error("Continue error:", error);
    abortTyping();
    replyElement.textContent = `Error: ${error?.message || error}`;
    setStatus("Continue failed");
  } finally {
    generating = false;
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
    console.error("Generation error:", error);
    abortTyping();
    replyElement.textContent = `Error: ${error?.message || error}`;
    setStatus("Generation failed");
  } finally {
    generating = false;
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
  try {
    restoreConversation();
    updateModelHeading();
    updateCost();
    updateMemory();
    await updateStorage();
    await loadCurrentModel();
  } catch (error) {
    console.error("STARTUP ERROR:", error);
    setStatus(`Startup error: ${error?.message || error}`);
  }
})();
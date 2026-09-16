import { Wllama, LoggerWithoutDebug } from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/index.js";
import WasmFromCDN from "https://cdn.jsdelivr.net/npm/@wllama/wllama@3.6.1/esm/wasm-from-cdn.js";

// ================== CONFIG ==================
const MODEL_URL =
  "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_0.gguf";

const THREADS = 4;        // tune: try 3, 4, 6
const N_CTX = 2048;
const N_PREDICT = 256;

const TYPE_MS = 30;       // ms per character — lower = faster typing feel

const COST_TOKEN_PER_1M = 0.20;
const COST_CPU_HOUR = 0.05;
const COST_KEY = "gemma-270m-wllama-cost-v1";
const CHAT_KEY = "gemma-270m-wllama-chat-v1";

// ================== TYPEWRITER ==================
// Letters appear one by one. The model generates at full speed in the
// background; this loop just paces the *display*. If the model outpaces
// the animation, it speeds up to catch up, so it never lags behind.
let typeEl = null;
let typeQueue = "";
let typeTimer = null;
let typeFinished = false;
let drainResolve = null;

function startTyping(el) {
  finishTyping(); // reset any previous state
  typeEl = el;
  typeQueue = "";
  typeFinished = false;
  el.classList.add("typing");
  typeTimer = setInterval(typeTick, TYPE_MS);
}

function typeTick() {
  if (!typeEl) return;
  if (!typeQueue.length) {
    if (typeFinished) finishTyping(); // all shown, generation done
    return;
  }
  // adaptive speed: 1 char/tick normally, more when there's a backlog
  let chars = 1;
  if (typeFinished) chars = 4;               // drain fast after generation ends
  else if (typeQueue.length > 40) chars = 3; // catching up
  else if (typeQueue.length > 15) chars = 2;
  typeEl.textContent += typeQueue.slice(0, chars);
  typeQueue = typeQueue.slice(chars);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function finishTyping() {
  if (typeTimer) { clearInterval(typeTimer); typeTimer = null; }
  if (typeEl) { typeEl.classList.remove("typing"); typeEl = null; }
  if (drainResolve) { drainResolve(); drainResolve = null; }
}

// resolves when generation is done AND all letters have been displayed
function typeFinish() {
  typeFinished = true;
  if (!typeTimer) return Promise.resolve(); // nothing running
  return new Promise((resolve) => { drainResolve = resolve; });
}

// hard stop (errors / clear button): drop pending text instantly
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
let totalCost = Number(localStorage.getItem(COST_KEY) || "0");

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

// ================== MODEL ==================
async function initializeModel() {
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  backendEl.textContent = isolated
    ? `wllama (llama.cpp) • ${THREADS} threads`
    : "wllama • 1 thread (isolation FAILED — check coi-serviceworker.js!)";

  setStatus("Loading Gemma 3 270M (Q4_0, 242 MB)…");
  setProgress(0);

  wllama = new Wllama(WasmFromCDN, { logger: LoggerWithoutDebug });

  await wllama.loadModelFromUrl(MODEL_URL, {
    n_ctx: N_CTX,
    n_threads: THREADS,
    progressCallback: ({ loaded, total }) => {
      setProgress((loaded / total) * 100);
      setStatus(`Downloading… ${Math.round((loaded / total) * 100)}%`);
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

  startTyping(replyElement);

  await wllama.createChatCompletion({
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
      typeQueue += piece; // animation loop picks it up
    },
  });

  await typeFinish(); // let the remaining letters finish appearing

  const ended = performance.now();
  const firstAt = firstTokenAt ?? ended, lastAt = lastTokenAt ?? ended;
  const prefillSeconds = Math.max((firstAt - started) / 1000, 0.001);
  const genSeconds = Math.max((lastAt - firstAt) / 1000, 0.001);

  // NOTE: speed is measured from real token arrival times, so the
  // typewriter animation never pollutes the tok/s number
  const outputTokens = Math.max(1, Math.round(generatedText.length / 4));
  const tokensPerSecond = outputTokens / genSeconds;

  speedEl.textContent =
    `${outputTokens} tok • ${tokensPerSecond.toFixed(1)} tok/s • prefill ${prefillSeconds.toFixed(1)}s`;

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
  await updateStorage();
  updateMemory();
}

// ================== SEND ==================
async function sendMessage() {
  if (generating) return;
  const prompt = inputEl.value.trim();
  if (!prompt) return;
  if (!wllama) { setStatus("Model is still loading…"); return; }
  generating = true;
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
    abortTyping(); // stop animation before overwriting with the error
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
    updateCost();
    updateMemory();
    await updateStorage();
    await initializeModel();
  } catch (error) {
    console.error("STARTUP ERROR:", error);
    setStatus(`Startup error: ${error?.message || error}`);
  }
})();
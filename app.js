import { env, pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

// ================== CONFIG ==================
const MODEL_ID = "onnx-community/SmolLM2-360M-Instruct-ONNX";
// Smarter but ~30% slower — swap in if you prefer quality over speed:
// const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct-ONNX";

const COST_TOKEN_PER_1M = 0.20;
const COST_CPU_HOUR = 0.05;
const COST_KEY = "smollm2-360m-total-cost-v3";
const CHAT_KEY = "smollm2-360m-chat-v3";

// ================== PERFORMANCE FIXES ==================
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.useWasmCache = true;

// FIX 1: multi-threaded WASM. Only takes effect when the page is
// cross-origin isolated — coi-serviceworker.js (loaded in index.html)
// handles that. Try 2 / 4 / 6 and keep whichever is fastest.
env.backends.onnx.wasm.numThreads = 4;

// FIX 2: run inference inside a Web Worker so the page
// doesn't freeze while generating.
env.backends.onnx.wasm.proxy = true;

// ================== ELEMENTS ==================
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), loadbarEl = $("loadbar"), loadfillEl = $("loadfill");
const backendEl = $("backend"), ramEl = $("ram"), storageEl = $("storage");
const speedEl = $("speed"), costEl = $("cost"), chatEl = $("chat");
const formEl = $("form"), inputEl = $("input"), sendEl = $("send"), clearEl = $("clear");

// ================== STATE ==================
let generator = null;
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

// ================== TOKEN COUNTING ==================
function estimateTokens(text) {
  const words = String(text).trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(words * 1.3));
}
function countTokens(text) {
  // real token count from the tokenizer, with a safe fallback
  try { return generator.tokenizer(text).input_ids.dims[1] || estimateTokens(text); }
  catch { return estimateTokens(text); }
}

// ================== MODEL ==================
async function initializeModel() {
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  backendEl.textContent = isolated
    ? `WASM/CPU • ${env.backends.onnx.wasm.numThreads} threads`
    : "WASM/CPU • 1 thread (isolation FAILED)";
  setStatus("Loading SmolLM2 360M…");
  setProgress(0);
  const progressCallback = (info) => {
    if (info?.status) setStatus(String(info.status));
    else if (info?.file) setStatus(`Loading ${info.file}…`);
    const p = Number(info?.progress);
    if (Number.isFinite(p)) setProgress(p);
  };
  // dtype "q4" — NOT q4f16, which hits a float16 WASM error on this setup
  generator = await pipeline("text-generation", MODEL_ID, {
    dtype: "q4",
    device: "wasm",
    progress_callback: progressCallback,
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
    { role: "system", content:
        "You are SmolLM2 360M, a small AI assistant running locally in the user's browser. " +
        "Do not claim to be another model or company. Answer directly and accurately. " +
        "For calculations, show your work." },
    ...conversation.slice(-6), // shorter history = much faster prefill on this chip
    { role: "user", content: prompt },
  ];

  const started = performance.now();
  let firstTokenAt = null, lastTokenAt = null, generatedText = "";

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    callback_function: (text) => {
      const now = performance.now();
      if (firstTokenAt === null) firstTokenAt = now; // FIX 3: timer starts at first token
      lastTokenAt = now;
      generatedText += text;
      replyElement.textContent = generatedText;
      chatEl.scrollTop = chatEl.scrollHeight;
    },
  });

  await generator(messages, {
    max_new_tokens: 256, // lower = snappier; raise for longer answers
    do_sample: true,
    temperature: 0.7,
    streamer,
  });

  const ended = performance.now();
  const firstAt = firstTokenAt ?? ended, lastAt = lastTokenAt ?? ended;
  const prefillSeconds = Math.max((firstAt - started) / 1000, 0.001);
  const genSeconds = Math.max((lastAt - firstAt) / 1000, 0.001);

  const outputTokens = countTokens(generatedText);
  const inputTokens = messages.reduce((t, m) => t + estimateTokens(m.content), 0);
  const totalTokens = inputTokens + outputTokens;
  const tokensPerSecond = outputTokens / genSeconds;

  speedEl.textContent =
    `${outputTokens} tok • ${tokensPerSecond.toFixed(1)} tok/s • prefill ${prefillSeconds.toFixed(1)}s`;

  const requestCost =
    (totalTokens / 1_000_000) * COST_TOKEN_PER_1M +
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
  if (!generator) { setStatus("Model is still loading…"); return; }
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
  conversation = [];
  localStorage.removeItem(CHAT_KEY);
  chatEl.replaceChildren();
});
setInterval(() => { updateMemory(); updateStorage(); }, 1500);

// ================== STARTUP ==================
// NOTE: no more custom sw.js registration — coi-serviceworker.js needs the
// service-worker slot (only one per scope), and transformers.js already
// caches the model itself via env.useBrowserCache.
async function main() {
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
}
main();
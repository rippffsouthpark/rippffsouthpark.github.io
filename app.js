import {
  env,
  pipeline,
  TextStreamer
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL_ID = "onnx-community/SmolLM2-360M-Instruct-ONNX";

// Cost assumptions are only rough equivalents, not real billing.
const COST_TOKEN_PER_1M = 0.20;
const COST_CPU_HOUR = 0.05;

const COST_KEY = "smollm2-360m-total-cost-v1";
const CHAT_KEY = "smollm2-360m-chat-v1";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.useWasmCache = true;

const $ = (id) => document.getElementById(id);

const statusEl = $("status");
const barEl = $("loadbar");
const fillEl = $("loadfill");
const chatEl = $("chat");
const inputEl = $("input");
const sendEl = $("send");
const ramEl = $("ram");
const storageEl = $("storage");
const speedEl = $("speed");
const costEl = $("cost");

let generator = null;
let conversation = [];
let generating = false;
let totalCost = Number(localStorage.getItem(COST_KEY) || "0");

function setStatus(s) { statusEl.textContent = s; }

function addMessage(role, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function updateMemory() {
  if (performance.memory) {
    const used = performance.memory.usedJSHeapSize / 1048576;
    const limit = performance.memory.jsHeapSizeLimit / 1048576;
    ramEl.textContent = `${used.toFixed(0)} / ${limit.toFixed(0)} MB`;
  } else {
    ramEl.textContent = "not exposed";
  }
}

async function updateStorage() {
  try {
    if (!navigator.storage?.estimate) {
      storageEl.textContent = "unavailable";
      return;
    }
    const s = await navigator.storage.estimate();
    const used = (s.usage || 0) / 1048576;
    const quota = (s.quota || 0) / 1048576;
    storageEl.textContent = quota
      ? `${used.toFixed(0)} / ${quota.toFixed(0)} MB`
      : `${used.toFixed(0)} MB`;
  } catch {
    storageEl.textContent = "unavailable";
  }
}

function updateCost() {
  costEl.textContent = `$${totalCost.toFixed(4)}`;
}

function saveConversation() {
  try {
    localStorage.setItem(CHAT_KEY, JSON.stringify(conversation.slice(-24)));
  } catch {}
}

function restoreConversation() {
  try {
    const saved = JSON.parse(localStorage.getItem(CHAT_KEY) || "[]");
    if (!Array.isArray(saved)) return;
    conversation = saved;
    for (const m of conversation) addMessage(m.role, m.content);
  } catch {}
}

async function init() {
  restoreConversation();
  updateCost();
  updateMemory();
  await updateStorage();

  barEl.style.display = "block";
  setStatus("Loading SmolLM2 360M…");

  const progress_callback = (info) => {
    const text = info?.status || info?.file || "Loading model…";
    setStatus(String(text));
    const p = Number(info?.progress);
    if (Number.isFinite(p)) {
      fillEl.style.width = `${Math.max(0, Math.min(100, p))}%`;
    }
  };

  generator = await pipeline(
    "text-generation",
    MODEL_ID,
    {
      dtype: "q4f16",
      device: "wasm",
      progress_callback
    }
  );

  fillEl.style.width = "100%";
  setTimeout(() => { barEl.style.display = "none"; }, 300);

  setStatus("Ready");
  inputEl.disabled = false;
  sendEl.disabled = false;
  inputEl.focus();
  await updateStorage();
}

function estimateTokens(text) {
  return Math.max(
    1,
    Math.ceil(
      String(text)
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .length * 1.3
    )
  );
}

async function generate(prompt, replyEl) {
  const messages = [
    {
      role: "system",
      content:
        "You are SmolLM2, a small AI assistant running locally in the browser. " +
        "Do not claim to be another company or model. " +
        "Answer directly and avoid inventing facts."
    },
    ...conversation.slice(-10),
    { role: "user", content: prompt }
  ];

  const started = performance.now();
  let generated = "";

  const streamer = new TextStreamer(
    generator.tokenizer,
    {
      skip_prompt: true,
      callback_function: (text) => {
        generated += text;
        replyEl.textContent = generated;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    }
  );

  await generator(messages, {
    max_new_tokens: 512,
    do_sample: true,
    temperature: 0.7,
    streamer
  });

  const elapsed = Math.max((performance.now() - started) / 1000, 0.001);
  const outputTokens = estimateTokens(generated);
  const inputTokens = messages.reduce(
    (sum, m) => sum + estimateTokens(m.content),
    0
  );
  const totalTokens = inputTokens + outputTokens;
  const tokps = outputTokens / elapsed;

  speedEl.textContent =
    `${outputTokens} tok • ${tokps.toFixed(1)} tok/s`;

  const tokenCost =
    (totalTokens / 1000000) * COST_TOKEN_PER_1M;

  const cpuCost =
    (elapsed / 3600) * COST_CPU_HOUR;

  const requestCost =
    tokenCost + cpuCost;

  totalCost += requestCost;
  localStorage.setItem(COST_KEY, String(totalCost));
  updateCost();

  conversation.push(
    { role: "user", content: prompt },
    { role: "assistant", content: generated }
  );
  saveConversation();
  await updateStorage();
  updateMemory();
}

async function sendMessage() {
  if (generating || !generator) return;

  const prompt = inputEl.value.trim();
  if (!prompt) return;

  generating = true;
  inputEl.value = "";
  inputEl.disabled = true;
  sendEl.disabled = true;

  addMessage("user", prompt);
  const replyEl = addMessage("assistant", "");

  try {
    setStatus("Generating…");
    await generate(prompt, replyEl);
    setStatus("Ready");
  } catch (error) {
    console.error(error);
    replyEl.textContent = `Error: ${error?.message || error}`;
    setStatus("Generation failed");
  } finally {
    generating = false;
    inputEl.disabled = false;
    sendEl.disabled = false;
    inputEl.focus();
  }
}

// Enter = send. Shift+Enter = newline.
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    event.stopPropagation();
    sendMessage();
  }
});

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

$("clear").addEventListener("click", () => {
  conversation = [];
  localStorage.removeItem(CHAT_KEY);
  chatEl.replaceChildren();
});

setInterval(() => {
  updateMemory();
  updateStorage();
}, 1500);

init().catch((error) => {
  console.error(error);
  setStatus(`Startup error: ${error?.message || error}`);
});

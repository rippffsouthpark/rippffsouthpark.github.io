import { pipeline, env, TextStreamer, LogLevel } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";
const COST_TOKEN_PER_1M = 0.20;
const COST_CPU_HOUR = 0.15;
const COST_KEY = "qwen-cpu-total-cost-v1";
const CHAT_KEY = "qwen-cpu-chat-v1";

// Force the CPU/WASM path. No WebGPU is requested or used.
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
env.useBrowserCache = true;
env.useWasmCache = true;
env.logLevel = LogLevel.ERROR;

const $ = id => document.getElementById(id);
const statusEl = $("status");
const loadbarEl = $("loadbar");
const loadfillEl = $("loadfill");
const ramEl = $("ram");
const storageEl = $("storage");
const speedEl = $("speed");
const costEl = $("cost");
const chatEl = $("chat");
const formEl = $("form");
const inputEl = $("input");
const sendEl = $("send");
const clearEl = $("clear");

let model = null;
let generating = false;
let conversation = [];
let totalCost = Number(localStorage.getItem(COST_KEY) || "0");

function setStatus(text){ statusEl.textContent = text; }
function setProgress(p){ loadbarEl.style.display="block"; loadfillEl.style.width=`${Math.max(0,Math.min(100,Number(p)||0))}%`; }
function hideProgress(){ loadbarEl.style.display="none"; loadfillEl.style.width="0%"; }
function addMessage(role,text=""){
  const el=document.createElement("div"); el.className=`msg ${role}`; el.textContent=text;
  chatEl.appendChild(el); chatEl.scrollTop=chatEl.scrollHeight; return el;
}
function updateCost(){ costEl.textContent=`$${totalCost.toFixed(4)}`; }
function updateMemory(){
  if(performance.memory){
    const used=performance.memory.usedJSHeapSize/1048576;
    const limit=performance.memory.jsHeapSizeLimit/1048576;
    ramEl.textContent=`${used.toFixed(0)} / ${limit.toFixed(0)} MB`;
  } else ramEl.textContent="not exposed";
}
async function updateStorage(){
  try{
    if(!navigator.storage?.estimate){storageEl.textContent="unavailable";return;}
    const s=await navigator.storage.estimate();
    const used=(s.usage||0)/1048576, quota=(s.quota||0)/1048576;
    storageEl.textContent=quota?`${used.toFixed(0)} / ${quota.toFixed(0)} MB`:`${used.toFixed(0)} MB`;
  }catch{storageEl.textContent="unavailable";}
}
function saveConversation(){ try{localStorage.setItem(CHAT_KEY,JSON.stringify(conversation.slice(-24)));}catch{} }
function restoreConversation(){
  try{
    const saved=JSON.parse(localStorage.getItem(CHAT_KEY)||"[]");
    if(!Array.isArray(saved))return; conversation=saved;
    for(const m of conversation)addMessage(m.role,m.content);
  }catch{}
}
async function registerSW(){
  if(!('serviceWorker' in navigator))return;
  try{ await navigator.serviceWorker.register('./sw.js',{scope:'./'}); }catch(e){ console.warn('SW:',e); }
}

async function initialize(){
  setStatus("Loading Qwen CPU runtime…");
  setProgress(0);
  model = await pipeline("text-generation", MODEL_ID, {
    device: "wasm",
    dtype: "q4",
    progress_callback: info => {
      const p = Number(info?.progress);
      if(Number.isFinite(p)){ setProgress(p); }
      if(info?.status === "progress" && info?.file){ setStatus(`Loading ${info.file}`); }
      else if(info?.status === "done"){ setStatus("Preparing CPU model…"); }
    },
  });
  hideProgress();
  setStatus("Ready • CPU / WASM");
  inputEl.disabled=false; sendEl.disabled=false; inputEl.focus();
  updateMemory(); await updateStorage(); updateCost();
}

function countTokensFallback(text){
  return Math.max(1, Math.ceil(text.trim().split(/\s+/).filter(Boolean).length * 1.3));
}

async function generate(prompt, replyEl){
  const messages=[
    {
      role:"system",
      content:"You are Qwen 2.5 0.5B, an AI assistant running locally on the user's CPU. Your name is Qwen. Do not claim to be Anthropic, OpenAI, Google, Claude, ChatGPT, or another company. Answer directly and accurately."
    },
    ...conversation.slice(-10),
    {role:"user",content:prompt}
  ];

  const started=performance.now();
  let streamedText="";
  let firstTokenTime=0;

  const streamer = new TextStreamer(model.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (!firstTokenTime) firstTokenTime = performance.now();
      streamedText += text;
      replyEl.textContent = streamedText;
      chatEl.scrollTop = chatEl.scrollHeight;
    }
  });

  const result = await model(messages, {
    max_new_tokens: 384,
    do_sample: true,
    temperature: 0.7,
    return_full_text: false,
    streamer
  });

  // Some streamer/model combinations return the final text too.
  const finalText = Array.isArray(result)
    ? String(result[0]?.generated_text || streamedText || "")
    : String(result?.generated_text || streamedText || "");

  if (finalText && finalText.length >= streamedText.length) {
    streamedText = finalText;
    replyEl.textContent = finalText;
  }

  const elapsed = Math.max((performance.now() - started) / 1000, 0.001);
  const outputTokens = countTokensFallback(streamedText);
  const inputTokens = countTokensFallback(prompt);
  const totalTokens = inputTokens + outputTokens;
  const tokps = outputTokens / elapsed;

  speedEl.textContent = `${outputTokens} tok • ${tokps.toFixed(1)} tok/s`;

  const tokenCost = (totalTokens / 1e6) * COST_TOKEN_PER_1M;
  const cpuCost = (elapsed / 3600) * COST_CPU_HOUR;
  totalCost += tokenCost + cpuCost;
  localStorage.setItem(COST_KEY, String(totalCost));
  updateCost();

  conversation.push(
    {role:"user",content:prompt},
    {role:"assistant",content:streamedText}
  );
  saveConversation();

  setStatus("Ready • CPU / WASM");
  updateMemory();
  await updateStorage();
}

async function sendMessage(){
  if(generating)return;
  const prompt=inputEl.value.trim();
  if(!prompt||!model)return;
  generating=true; inputEl.value=""; inputEl.disabled=true; sendEl.disabled=true;
  addMessage("user",prompt);
  const reply=addMessage("assistant","");
  try{ setStatus("Generating…"); await generate(prompt,reply); }
  catch(e){ console.error(e); reply.textContent=`Error: ${e?.message||e}`; setStatus("Generation failed"); }
  finally{ generating=false; inputEl.disabled=false; sendEl.disabled=false; inputEl.focus(); }
}

inputEl.addEventListener("keydown", e => {
  if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); e.stopPropagation(); sendMessage(); }
});
formEl.addEventListener("submit", e => { e.preventDefault(); sendMessage(); });
clearEl.addEventListener("click", () => { conversation=[]; localStorage.removeItem(CHAT_KEY); chatEl.replaceChildren(); });
setInterval(() => { updateMemory(); updateStorage(); },1500);

(async()=>{
  try{ restoreConversation(); updateCost(); updateMemory(); await updateStorage(); await registerSW(); await initialize(); }
  catch(e){ console.error("STARTUP ERROR:",e); setStatus(`Startup error: ${e?.message||e}`); hideProgress(); }
})();

import { BrowserAI } from "https://cdn.jsdelivr.net/npm/@browserai/browserai/+esm";

const MODEL_ID = "qwen2.5-0.5b-flare";
const COST_KEY = "qwen-local-equivalent-spend-v1";
const CHAT_KEY = "qwen-local-chat-v1";
const RUNS_KEY = "qwen-local-runs-v1";

// Adjustable estimates. These are NOT an actual provider bill.
const TOKEN_EQUIVALENT_USD_PER_1M = 0.20;
const GPU_EQUIVALENT_USD_PER_HOUR = 1.00;

const $ = id => document.getElementById(id);
const statusEl = $("status"), fillEl = $("loadfill"), barEl = $("loadbar"), chatEl = $("chat");
const inputEl = $("input"), sendEl = $("send"), backendEl = $("backend"), ramEl = $("ram");
const storageEl = $("storage"), speedEl = $("speed"), costEl = $("cost");

let ai = null;
let conversation = [];
let totalEquivalentSpend = Number(localStorage.getItem(COST_KEY) || 0);
let runs = Number(localStorage.getItem(RUNS_KEY) || 0);
costEl.textContent = `$${totalEquivalentSpend.toFixed(4)}`;

function status(s){ statusEl.textContent = s; }
function mb(n){ return Number.isFinite(n) ? `${n.toFixed(0)} MB` : "—"; }
function addMessage(role,text=""){
  const el=document.createElement("div"); el.className=`msg ${role}`; el.textContent=text; chatEl.appendChild(el); chatEl.scrollTop=chatEl.scrollHeight; return el;
}
function save(){ localStorage.setItem(CHAT_KEY, JSON.stringify(conversation.slice(-40))); localStorage.setItem(COST_KEY,String(totalEquivalentSpend)); localStorage.setItem(RUNS_KEY,String(runs)); }
function restore(){ try{const x=JSON.parse(localStorage.getItem(CHAT_KEY)||"[]"); if(Array.isArray(x)){conversation=x; for(const m of x)addMessage(m.role,m.content)}}catch{}}
async function metrics(){
  if(performance.memory) ramEl.textContent=`${mb(performance.memory.usedJSHeapSize/1048576)} / ${mb(performance.memory.jsHeapSizeLimit/1048576)}`; else ramEl.textContent="not exposed";
  try{if(navigator.storage?.estimate){const s=await navigator.storage.estimate();storageEl.textContent=`${mb((s.usage||0)/1048576)} / ${mb((s.quota||0)/1048576)}`}}catch{}
}
function normalizeText(x){
  if(typeof x === "string") return x;
  return x?.text ?? x?.choices?.[0]?.message?.content ?? x?.output ?? "";
}
function normalizeStats(x){ return x?.stats ?? x?.usage ?? x?.metrics ?? {}; }
function estimateTokens(text){
  const words=text.trim()?text.trim().split(/\s+/).filter(Boolean).length:0;
  return Math.max(0,Math.ceil(words*1.3));
}
async function init(){
  restore(); await metrics();
  backendEl.textContent = navigator.gpu ? "WebGPU" : "Unavailable";
  if(!navigator.gpu) throw new Error("WebGPU is unavailable. ChromeOS policy or this browser may have it disabled.");

  status("Loading BrowserAI runtime…");
  ai = new BrowserAI({modelSource:"direct",cacheBackend:"indexeddb"});
  ai.on?.("loadprogress", ({progress,status:s})=>{
    barEl.style.display="block"; fillEl.style.width=`${Math.max(0,Math.min(100,Number(progress)||0))}%`;
    if(s) status(String(s));
  });
  status("Checking local model cache…");
  await ai.loadModel(MODEL_ID,{onProgress:({progress,status:s})=>{
    barEl.style.display="block"; fillEl.style.width=`${Math.max(0,Math.min(100,Number(progress)||0))}%`; if(s)status(String(s));
  }});
  barEl.style.display="none"; status("Qwen ready — cached after the first download.");
  inputEl.disabled=false; sendEl.disabled=false; inputEl.focus();
}
async function run(){
  const prompt=inputEl.value.trim(); if(!prompt||!ai)return;
  inputEl.value=""; inputEl.disabled=true; sendEl.disabled=true;
  addMessage("user",prompt); const reply=addMessage("assistant","");
  const started=performance.now(); status("Generating locally…");
  const messages=[{role:"system",content:"You are a helpful, direct assistant."},...conversation.slice(-20),{role:"user",content:prompt}];
  try{
    const result=await ai.generateText(messages,{
      temperature:0.7,max_tokens:512,runtime:{jsonMode:"none"},
      onDelta:full=>{const t=normalizeText(full);reply.textContent=t;chatEl.scrollTop=chatEl.scrollHeight;}
    });
    const text=normalizeText(result); reply.textContent=text;
    const elapsed=Math.max((performance.now()-started)/1000,0.001);
    const stats=normalizeStats(result);
    const inTok=Number(stats.prompt_tokens ?? stats.input_tokens ?? stats.promptTokens ?? 0) || estimateTokens(prompt);
    const outTok=Number(stats.completion_tokens ?? stats.output_tokens ?? stats.completionTokens ?? 0) || estimateTokens(text);
    const totalTok=Number(stats.total_tokens ?? stats.totalTokens ?? (inTok+outTok));
    const providerTokenCost=totalTok/1e6*TOKEN_EQUIVALENT_USD_PER_1M;
    const equivalentComputeCost=elapsed/3600*GPU_EQUIVALENT_USD_PER_HOUR;
    const estimated=providerTokenCost+equivalentComputeCost;
    totalEquivalentSpend+=estimated; runs++; save();
    const tps=outTok/elapsed;
    speedEl.textContent=`${outTok} tok • ${tps.toFixed(1)} tok/s`;
    costEl.textContent=`$${totalEquivalentSpend.toFixed(4)}`;
    status(`Done • ${inTok} in / ${outTok} out • est $${estimated.toFixed(6)}`);
    conversation.push({role:"user",content:prompt},{role:"assistant",content:text}); save();
    await metrics();
  }catch(e){console.error(e);reply.textContent=`Error: ${e?.message||e}`;status("Generation failed.")}
  finally{inputEl.disabled=false;sendEl.disabled=false;inputEl.focus()}
}
$("form").addEventListener("submit",e=>{e.preventDefault();run()});
$("clear").addEventListener("click",()=>{conversation=[];localStorage.removeItem(CHAT_KEY);chatEl.replaceChildren()});
setInterval(metrics,1500);
init().catch(e=>{console.error(e);status(`Startup error: ${e?.message||e}`);inputEl.disabled=true;sendEl.disabled=true});

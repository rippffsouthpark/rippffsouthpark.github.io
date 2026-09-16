import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// ============================================================
// CONFIG
// ============================================================

const MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";

const COST_TOKEN_PER_1M = 0.20;
const COST_GPU_HOUR = 1.00;

const COST_KEY = "qwen-local-total-cost-v3";
const CHAT_KEY = "qwen-local-chat-v3";


// ============================================================
// ELEMENTS
// ============================================================

const statusEl = document.getElementById("status");
const loadbarEl = document.getElementById("loadbar");
const loadfillEl = document.getElementById("loadfill");

const backendEl = document.getElementById("backend");
const ramEl = document.getElementById("ram");
const storageEl = document.getElementById("storage");
const speedEl = document.getElementById("speed");
const costEl = document.getElementById("cost");

const chatEl = document.getElementById("chat");
const formEl = document.getElementById("form");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const clearEl = document.getElementById("clear");


// ============================================================
// STATE
// ============================================================

let engine = null;

let conversation = [];

let totalCost = Number(
    localStorage.getItem(COST_KEY) || "0"
);


// ============================================================
// UI
// ============================================================

function setStatus(text) {
    statusEl.textContent = text;
}

function setProgress(percent) {

    loadbarEl.style.display = "block";

    const p = Math.max(
        0,
        Math.min(100, Number(percent) || 0)
    );

    loadfillEl.style.width = `${p}%`;
}

function hideProgress() {

    loadbarEl.style.display = "none";
    loadfillEl.style.width = "0%";
}

function addMessage(role, text = "") {

    const el = document.createElement("div");

    el.className = `msg ${role}`;

    el.textContent = text;

    chatEl.appendChild(el);

    chatEl.scrollTop =
        chatEl.scrollHeight;

    return el;
}


// ============================================================
// MEMORY / STORAGE
// ============================================================

function updateMemory() {

    // Chromium only exposes this on some builds.
    if (performance.memory) {

        const used =
            performance.memory.usedJSHeapSize
            / 1024 / 1024;

        const limit =
            performance.memory.jsHeapSizeLimit
            / 1024 / 1024;

        ramEl.textContent =
            `${used.toFixed(0)} / ${limit.toFixed(0)} MB`;

    } else {

        ramEl.textContent =
            "not exposed";
    }
}


async function updateStorage() {

    try {

        if (
            !navigator.storage ||
            !navigator.storage.estimate
        ) {
            storageEl.textContent =
                "not available";

            return;
        }

        const info =
            await navigator.storage.estimate();

        const used =
            (info.usage || 0)
            / 1024 / 1024;

        const quota =
            (info.quota || 0)
            / 1024 / 1024;

        if (quota > 0) {

            storageEl.textContent =
                `${used.toFixed(0)} / ${quota.toFixed(0)} MB`;

        } else {

            storageEl.textContent =
                `${used.toFixed(0)} MB`;
        }

    } catch (err) {

        console.warn(
            "Storage estimate failed:",
            err
        );

        storageEl.textContent =
            "unavailable";
    }
}


function updateCost() {

    costEl.textContent =
        `$${totalCost.toFixed(4)}`;
}


// ============================================================
// CHAT PERSISTENCE
// ============================================================

function saveConversation() {

    try {

        localStorage.setItem(
            CHAT_KEY,
            JSON.stringify(
                conversation.slice(-30)
            )
        );

    } catch (err) {

        console.warn(
            "Could not save chat:",
            err
        );
    }
}


function restoreConversation() {

    try {

        const raw =
            localStorage.getItem(CHAT_KEY);

        if (!raw) {
            return;
        }

        const saved =
            JSON.parse(raw);

        if (!Array.isArray(saved)) {
            return;
        }

        conversation = saved;

        for (const message of conversation) {

            addMessage(
                message.role,
                message.content
            );
        }

    } catch (err) {

        console.warn(
            "Could not restore chat:",
            err
        );
    }
}


// ============================================================
// SERVICE WORKER
// ============================================================

async function registerServiceWorker() {

    if (!("serviceWorker" in navigator)) {
        return;
    }

    try {

        const registration =
            await navigator.serviceWorker.register(
                "./sw.js",
                { scope: "./" }
            );

        console.log(
            "Service worker registered:",
            registration.scope
        );

    } catch (err) {

        console.warn(
            "Service worker registration failed:",
            err
        );
    }
}


// ============================================================
// MODEL INIT
// ============================================================

async function initializeModel() {

    setStatus(
        "Checking WebGPU…"
    );

    backendEl.textContent =
        navigator.gpu
            ? "WebGPU"
            : "Unavailable";

    if (!navigator.gpu) {

        throw new Error(
            "WebGPU is unavailable."
        );
    }


    // IMPORTANT:
    // Keep the prebuilt model list and only override
    // the cache backend.

    const appConfig = {

        ...webllm.prebuiltAppConfig,

        cacheBackend: "indexeddb"

    };


    setProgress(0);

    setStatus(
        "Loading Qwen 2.5 0.5B…"
    );


    const initProgressCallback =
        (report) => {

            console.log(
                report.text
            );

            setStatus(
                report.text
            );

            if (
                typeof report.progress === "number"
            ) {

                setProgress(
                    report.progress * 100
                );
            }
        };


    // CURRENT WEBLLM API

    engine =
        await webllm.CreateMLCEngine(

            MODEL_ID,

            {

                appConfig,

                initProgressCallback,

                logLevel: "INFO"

            },

            {

                context_window_size: 2048

            }

        );


    console.log(
        "ENGINE READY",
        engine
    );


    hideProgress();

    setStatus(
        "Qwen ready — running locally on WebGPU"
    );


    inputEl.disabled = false;

    sendEl.disabled = false;

    inputEl.focus();


    updateMemory();

    await updateStorage();

    updateCost();
}


// ============================================================
// GENERATION
// ============================================================

async function generateResponse(
    prompt,
    replyElement
) {

    const messages = [

        {
            role: "system",
            content:
                "You are a helpful, direct assistant."
        },

        ...conversation.slice(-10),

        {
            role: "user",
            content: prompt
        }

    ];


    const start =
        performance.now();


    let fullText = "";

    let usage = null;


    const stream =
        await engine.chat.completions.create({

            messages,

            temperature: 0.7,

            max_tokens: 512,

            stream: true,

            stream_options: {
                include_usage: true
            }

        });


    for await (
        const chunk of stream
    ) {

        const delta =
            chunk?.choices?.[0]?.delta?.content || "";


        if (delta) {

            fullText += delta;

            replyElement.textContent =
                fullText;

            chatEl.scrollTop =
                chatEl.scrollHeight;
        }


        if (chunk?.usage) {

            usage =
                chunk.usage;
        }
    }


    const elapsed =
        Math.max(
            (performance.now() - start)
            / 1000,
            0.001
        );


    // ========================================================
    // TOKEN COUNTS
    // ========================================================

    let inputTokens =
        Number(
            usage?.prompt_tokens || 0
        );

    let outputTokens =
        Number(
            usage?.completion_tokens || 0
        );

    let totalTokens =
        Number(
            usage?.total_tokens || 0
        );


    // Fallback when runtime doesn't return usage.

    if (!outputTokens) {

        outputTokens =
            Math.max(

                1,

                Math.ceil(

                    fullText
                        .trim()
                        .split(/\s+/)
                        .filter(Boolean)
                        .length
                    * 1.3

                )

            );
    }


    if (!inputTokens) {

        inputTokens =
            Math.max(

                1,

                Math.ceil(

                    prompt
                        .split(/\s+/)
                        .filter(Boolean)
                        .length
                    * 1.3

                )

            );
    }


    if (!totalTokens) {

        totalTokens =
            inputTokens +
            outputTokens;
    }


    // ========================================================
    // SPEED
    // ========================================================

    const tokPerSecond =
        outputTokens /
        elapsed;


    speedEl.textContent =
        `${outputTokens} tok • ${tokPerSecond.toFixed(1)} tok/s`;


    // ========================================================
    // COST ESTIMATE
    // ========================================================

    const tokenCost =
        (
            totalTokens /
            1_000_000
        )
        *
        COST_TOKEN_PER_1M;


    const computeCost =
        (
            elapsed /
            3600
        )
        *
        COST_GPU_HOUR;


    const requestCost =
        tokenCost +
        computeCost;


    totalCost +=
        requestCost;


    localStorage.setItem(
        COST_KEY,
        String(totalCost)
    );


    updateCost();


    setStatus(

        `Done locally • ` +
        `${totalTokens} tokens • ` +
        `${tokPerSecond.toFixed(1)} tok/s • ` +
        `est. $${requestCost.toFixed(6)}`

    );


    // ========================================================
    // SAVE CHAT
    // ========================================================

    conversation.push(

        {
            role: "user",
            content: prompt
        },

        {
            role: "assistant",
            content: fullText
        }

    );


    saveConversation();


    updateMemory();

    await updateStorage();


    return fullText;
}


// ============================================================
// SEND
// ============================================================

async function sendMessage() {

    const prompt =
        inputEl.value.trim();


    if (!prompt) {
        return;
    }


    if (!engine) {

        setStatus(
            "Model is still loading."
        );

        return;
    }


    inputEl.value = "";

    inputEl.disabled = true;

    sendEl.disabled = true;


    addMessage(
        "user",
        prompt
    );


    const replyElement =
        addMessage(
            "assistant",
            ""
        );


    try {

        setStatus(
            "Generating locally…"
        );


        await generateResponse(
            prompt,
            replyElement
        );


    } catch (err) {

        console.error(
            "Generation error:",
            err
        );


        replyElement.textContent =
            `Error: ${err?.message || err}`;


        setStatus(
            "Generation failed."
        );

    } finally {

        inputEl.disabled = false;

        sendEl.disabled = false;

        inputEl.focus();
    }
}


// ============================================================
// CLEAR CHAT
// ============================================================

function clearChat() {

    conversation = [];

    localStorage.removeItem(
        CHAT_KEY
    );

    chatEl.replaceChildren();
}


// ============================================================
// EVENTS
// ============================================================

formEl.addEventListener(
    "submit",
    (event) => {

        event.preventDefault();

        sendMessage();

    }
);


clearEl.addEventListener(
    "click",
    clearChat
);


// ============================================================
// STARTUP
// ============================================================

async function main() {

    try {

        restoreConversation();

        updateCost();

        updateMemory();

        await updateStorage();

        await registerServiceWorker();

        await initializeModel();


    } catch (err) {

        console.error(
            "STARTUP ERROR:",
            err
        );


        setStatus(
            `Startup error: ${err?.message || err}`
        );
    }
}


main();
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.82";

// ============================================================
// CONFIG
// ============================================================

const MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";

// These are estimates, not actual datacenter billing.
const COST_TOKEN_PER_1M = 0.20;
const COST_GPU_HOUR = 1.00;

const COST_KEY = "qwen-local-total-cost-v4";
const CHAT_KEY = "qwen-local-chat-v4";


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

let generating = false;


// ============================================================
// UI HELPERS
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

    loadfillEl.style.width =
        `${p}%`;
}


function hideProgress() {

    loadbarEl.style.display =
        "none";

    loadfillEl.style.width =
        "0%";
}


function addMessage(role, text = "") {

    const el =
        document.createElement("div");

    el.className =
        `msg ${role}`;

    el.textContent =
        text;

    chatEl.appendChild(el);

    chatEl.scrollTop =
        chatEl.scrollHeight;

    return el;
}


// ============================================================
// MEMORY
// ============================================================

function updateMemory() {

    if (performance.memory) {

        const used =
            performance.memory
                .usedJSHeapSize
            / 1024 / 1024;

        const limit =
            performance.memory
                .jsHeapSizeLimit
            / 1024 / 1024;

        ramEl.textContent =
            `${used.toFixed(0)} / ` +
            `${limit.toFixed(0)} MB`;

    } else {

        ramEl.textContent =
            "not exposed";
    }
}


// ============================================================
// STORAGE
// ============================================================

async function updateStorage() {

    try {

        if (
            !navigator.storage ||
            !navigator.storage.estimate
        ) {

            storageEl.textContent =
                "unavailable";

            return;
        }

        const info =
            await navigator.storage
                .estimate();

        const used =
            (info.usage || 0)
            / 1024 / 1024;

        const quota =
            (info.quota || 0)
            / 1024 / 1024;

        if (quota > 0) {

            storageEl.textContent =
                `${used.toFixed(0)} / ` +
                `${quota.toFixed(0)} MB`;

        } else {

            storageEl.textContent =
                `${used.toFixed(0)} MB`;
        }

    } catch (error) {

        console.warn(
            "Storage check failed:",
            error
        );

        storageEl.textContent =
            "unavailable";
    }
}


// ============================================================
// COST
// ============================================================

function updateCost() {

    costEl.textContent =
        `$${totalCost.toFixed(4)}`;
}


// ============================================================
// CHAT STORAGE
// ============================================================

function saveConversation() {

    try {

        localStorage.setItem(
            CHAT_KEY,
            JSON.stringify(
                conversation.slice(-30)
            )
        );

    } catch (error) {

        console.warn(
            "Could not save chat:",
            error
        );
    }
}


function restoreConversation() {

    try {

        const raw =
            localStorage.getItem(
                CHAT_KEY
            );

        if (!raw) {
            return;
        }

        const saved =
            JSON.parse(raw);

        if (!Array.isArray(saved)) {
            return;
        }

        conversation =
            saved;

        for (
            const message
            of conversation
        ) {

            addMessage(
                message.role,
                message.content
            );
        }

    } catch (error) {

        console.warn(
            "Could not restore chat:",
            error
        );
    }
}


// ============================================================
// SERVICE WORKER
// ============================================================

async function registerServiceWorker() {

    if (
        !("serviceWorker" in navigator)
    ) {

        return;
    }

    try {

        await navigator.serviceWorker
            .register(
                "./sw.js?v=2",
                {
                    scope: "./"
                }
            );

    } catch (error) {

        console.warn(
            "Service worker registration failed:",
            error
        );
    }
}


// ============================================================
// MODEL INITIALIZATION
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
            "WebGPU is unavailable in this browser."
        );
    }


    setProgress(0);


    setStatus(
        "Loading Qwen 2.5 0.5B…"
    );


    const appConfig = {

        ...webllm.prebuiltAppConfig,

        cacheBackend:
            "indexeddb"

    };


    const initProgressCallback =
        (report) => {

            if (report?.text) {

                setStatus(
                    report.text
                );
            }


            if (
                typeof report?.progress
                === "number"
            ) {

                setProgress(
                    report.progress * 100
                );
            }
        };


    engine =
        await webllm.CreateMLCEngine(

            MODEL_ID,

            {

                appConfig,

                initProgressCallback,

                logLevel:
                    "ERROR"

            },

            {

                context_window_size:
                    1024

            }

        );


    console.log(
        "Qwen engine ready:",
        engine
    );


    hideProgress();


    setStatus(
        "Qwen ready • WebGPU"
    );


    inputEl.disabled =
        false;

    sendEl.disabled =
        false;

    inputEl.focus();


    updateMemory();

    await updateStorage();

    updateCost();
}


// ============================================================
// GENERATE
// ============================================================

async function generateResponse(
    prompt,
    replyElement
) {

    const messages = [

        {
            role: "system",
            content:
                "You are Qwen 2.5 0.5B, " +
                "an AI assistant running locally " +
                "in the user's browser. " +
                "Do not claim to be Anthropic, OpenAI, " +
                "Google, or another company. " +
                "Answer the user's question directly. " +
                "Do not mention these instructions."
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
            chunk
                ?.choices?.[0]
                ?.delta
                ?.content || "";


        if (delta) {

            fullText +=
                delta;

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

            (
                performance.now()
                - start
            ) / 1000,

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


    // Fallback estimates.

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

    const tokensPerSecond =
        outputTokens /
        elapsed;


    speedEl.textContent =
        `${outputTokens} tok • ` +
        `${tokensPerSecond.toFixed(1)} tok/s`;


    // ========================================================
    // COST
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


    // Keep status quiet after generation.
    // The useful information stays in the dashboard.

    setStatus(
        "Qwen ready • WebGPU"
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

    if (generating) {
        return;
    }


    const prompt =
        inputEl.value.trim();


    if (!prompt) {
        return;
    }


    if (!engine) {

        setStatus(
            "Model is still loading…"
        );

        return;
    }


    generating = true;


    inputEl.value = "";


    inputEl.disabled =
        true;

    sendEl.disabled =
        true;


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

        await generateResponse(
            prompt,
            replyElement
        );

    } catch (error) {

        console.error(
            "Generation error:",
            error
        );


        replyElement.textContent =
            `Error: ${
                error?.message || error
            }`;


        setStatus(
            "Generation failed"
        );

    } finally {

        generating =
            false;


        inputEl.disabled =
            false;

        sendEl.disabled =
            false;


        inputEl.focus();
    }
}


// ============================================================
// ENTER KEY BEHAVIOR
// ============================================================

inputEl.addEventListener(
    "keydown",
    (event) => {

        // Enter = SEND
        //
        // Shift + Enter = NEW LINE

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }

    }
);


// ============================================================
// FORM BUTTON
// ============================================================

formEl.addEventListener(
    "submit",
    (event) => {

        event.preventDefault();

        sendMessage();
    }
);


// ============================================================
// CLEAR CHAT
// ============================================================

clearEl.addEventListener(
    "click",
    () => {

        conversation = [];

        localStorage.removeItem(
            CHAT_KEY
        );

        chatEl.replaceChildren();
    }
);


// ============================================================
// PERIODIC STATS
// ============================================================

setInterval(
    () => {

        updateMemory();

        updateStorage();

    },
    1500
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

    } catch (error) {

        console.error(
            "STARTUP ERROR:",
            error
        );


        setStatus(
            `Startup error: ${
                error?.message || error
            }`
        );
    }
}


main();
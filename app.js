import {
    env,
    pipeline,
    TextStreamer
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


// ============================================================
// CONFIG
// ============================================================

const MODEL_ID =
    "onnx-community/SmolLM2-360M-Instruct-ONNX";

const COST_TOKEN_PER_1M =
    0.20;

const COST_CPU_HOUR =
    0.05;

const COST_KEY =
    "smollm2-360m-total-cost-v3";

const CHAT_KEY =
    "smollm2-360m-chat-v3";


// ============================================================
// TRANSFORMERS.JS
// ============================================================

env.allowLocalModels =
    false;

env.allowRemoteModels =
    true;

env.useBrowserCache =
    true;

env.useWasmCache =
    true;


// ============================================================
// ELEMENTS
// ============================================================

const statusEl =
    document.getElementById("status");

const loadbarEl =
    document.getElementById("loadbar");

const loadfillEl =
    document.getElementById("loadfill");

const backendEl =
    document.getElementById("backend");

const ramEl =
    document.getElementById("ram");

const storageEl =
    document.getElementById("storage");

const speedEl =
    document.getElementById("speed");

const costEl =
    document.getElementById("cost");

const chatEl =
    document.getElementById("chat");

const formEl =
    document.getElementById("form");

const inputEl =
    document.getElementById("input");

const sendEl =
    document.getElementById("send");

const clearEl =
    document.getElementById("clear");


// ============================================================
// STATE
// ============================================================

let generator =
    null;

let conversation =
    [];

let generating =
    false;

let totalCost =
    Number(
        localStorage.getItem(
            COST_KEY
        ) || "0"
    );


// ============================================================
// UI
// ============================================================

function setStatus(text) {
    statusEl.textContent =
        text;
}


function setProgress(percent) {

    loadbarEl.style.display =
        "block";

    const value =
        Math.max(
            0,
            Math.min(
                100,
                Number(percent) || 0
            )
        );

    loadfillEl.style.width =
        `${value}%`;
}


function hideProgress() {

    loadbarEl.style.display =
        "none";

    loadfillEl.style.width =
        "0%";
}


function addMessage(
    role,
    text = ""
) {

    const element =
        document.createElement(
            "div"
        );

    element.className =
        `msg ${role}`;

    element.textContent =
        text;

    chatEl.appendChild(
        element
    );

    chatEl.scrollTop =
        chatEl.scrollHeight;

    return element;
}


// ============================================================
// MEMORY
// ============================================================

function updateMemory() {

    if (
        performance.memory
    ) {

        const used =
            performance
                .memory
                .usedJSHeapSize
            / 1024
            / 1024;

        const limit =
            performance
                .memory
                .jsHeapSizeLimit
            / 1024
            / 1024;

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
            / 1024
            / 1024;

        const quota =
            (info.quota || 0)
            / 1024
            / 1024;

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
            "Storage estimate failed:",
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
                conversation.slice(-24)
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
            JSON.parse(
                raw
            );

        if (
            !Array.isArray(saved)
        ) {

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
                "./sw.js?v=5",
                {
                    scope:
                        "./"
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

    backendEl.textContent =
        "WASM / CPU";


    setStatus(
        "Loading SmolLM2 360M…"
    );


    setProgress(
        0
    );


    const progressCallback =
        (info) => {

            console.log(
                info
            );


            if (
                info?.status
            ) {

                setStatus(
                    String(
                        info.status
                    )
                );

            } else if (
                info?.file
            ) {

                setStatus(
                    `Loading ${
                        info.file
                    }…`
                );
            }


            const progress =
                Number(
                    info?.progress
                );


            if (
                Number.isFinite(
                    progress
                )
            ) {

                setProgress(
                    progress
                );
            }
        };


    // ========================================================
    // IMPORTANT
    // ========================================================
    //
    // q4 instead of q4f16.
    //
    // Your previous q4f16 runtime reached inference but
    // ONNX Runtime reported:
    //
    // Actual: tensor(float16)
    // Expected: tensor(float)
    //
    // q4 avoids that float16 path for this WASM setup.
    // ========================================================

    generator =
        await pipeline(

            "text-generation",

            MODEL_ID,

            {

                dtype:
                    "q4",

                device:
                    "wasm",

                progress_callback:
                    progressCallback
            }

        );


    hideProgress();


    setStatus(
        "Ready"
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
// TOKEN ESTIMATION
// ============================================================

function estimateTokens(
    text
) {

    const words =
        String(
            text
        )
            .trim()
            .split(
                /\s+/
            )
            .filter(
                Boolean
            )
            .length;


    return Math.max(
        1,
        Math.ceil(
            words * 1.3
        )
    );
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
            role:
                "system",

            content:
                "You are SmolLM2 360M, " +
                "a small AI assistant running locally " +
                "in the user's browser. " +
                "Do not claim to be another model or company. " +
                "Answer directly and accurately. " +
                "For calculations, show your work."
        },

        ...conversation.slice(
            -10
        ),

        {
            role:
                "user",

            content:
                prompt
        }

    ];


    const started =
        performance.now();


    let generatedText =
        "";


    const streamer =
        new TextStreamer(

            generator.tokenizer,

            {

                skip_prompt:
                    true,

                callback_function:
                    (
                        text
                    ) => {

                        generatedText +=
                            text;

                        replyElement.textContent =
                            generatedText;

                        chatEl.scrollTop =
                            chatEl.scrollHeight;
                    }
            }
        );


    await generator(

        messages,

        {

            max_new_tokens:
                512,

            do_sample:
                true,

            temperature:
                0.7,

            streamer
        }

    );


    const elapsed =
        Math.max(

            (
                performance.now()
                - started
            )
            / 1000,

            0.001
        );


    // ========================================================
    // TOKENS
    // ========================================================

    const outputTokens =
        estimateTokens(
            generatedText
        );


    const inputTokens =
        messages.reduce(

            (
                total,
                message
            ) =>

                total +
                estimateTokens(
                    message.content
                ),

            0
        );


    const totalTokens =
        inputTokens +
        outputTokens;


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
    // ESTIMATED COST
    // ========================================================

    const tokenCost =
        (
            totalTokens /
            1_000_000
        )
        *
        COST_TOKEN_PER_1M;


    const cpuCost =
        (
            elapsed /
            3600
        )
        *
        COST_CPU_HOUR;


    const requestCost =
        tokenCost +
        cpuCost;


    totalCost +=
        requestCost;


    localStorage.setItem(
        COST_KEY,
        String(
            totalCost
        )
    );


    updateCost();


    // ========================================================
    // SAVE CONVERSATION
    // ========================================================

    conversation.push(

        {
            role:
                "user",

            content:
                prompt
        },

        {
            role:
                "assistant",

            content:
                generatedText
        }

    );


    saveConversation();


    await updateStorage();

    updateMemory();
}


// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage() {

    if (
        generating
    ) {

        return;
    }


    const prompt =
        inputEl.value.trim();


    if (!prompt) {

        return;
    }


    if (!generator) {

        setStatus(
            "Model is still loading…"
        );

        return;
    }


    generating =
        true;


    inputEl.value =
        "";


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

        setStatus(
            "Generating…"
        );


        await generateResponse(
            prompt,
            replyElement
        );


        setStatus(
            "Ready"
        );


    } catch (error) {

        console.error(
            "Generation error:",
            error
        );


        replyElement.textContent =
            `Error: ${
                error?.message ||
                error
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
// ENTER = SEND
// SHIFT+ENTER = NEW LINE
// ============================================================

inputEl.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            event.stopPropagation();

            sendMessage();
        }

    }
);


// ============================================================
// FORM
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

        conversation =
            [];

        localStorage.removeItem(
            CHAT_KEY
        );

        chatEl.replaceChildren();
    }
);


// ============================================================
// LIVE STATS
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
                error?.message ||
                error
            }`
        );
    }
}


main();

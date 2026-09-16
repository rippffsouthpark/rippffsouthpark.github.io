# Qwen 2.5 0.5B — Local Chromebook Browser App

Uses the current BrowserAI Flare registry entry `qwen2.5-0.5b-flare`, which is the Qwen2.5-0.5B-Instruct Q4_K_M GGUF.

## Features
- WebGPU-first local inference
- BrowserAI model artifact cache (IndexedDB)
- Service-worker cache for the small app shell
- Streaming responses
- Chat history persisted locally
- JS heap display when Chromium exposes `performance.memory`
- Origin storage usage/quota
- Output token estimates and tok/s
- Persistent all-time equivalent-spend estimate

## Important
Open this through GitHub Pages (HTTPS), not `file://`.

The cost display is an estimate only. It uses editable proxy rates in `app.js`:
`TOKEN_EQUIVALENT_USD_PER_1M = 0.20`
`GPU_EQUIVALENT_USD_PER_HOUR = 1.00`

It does not represent the actual cost of a specific AI provider or datacenter.

Qwen2.5-0.5B-Instruct is an instruction-tuned model. This browser wrapper does not remove its training-time behavioral safeguards.

The model is separate from this repository/app and is fetched by BrowserAI from its model source on first load, then cached by the browser.

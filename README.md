# SmolLM2 360M — Chromebook CPU version

This is the CPU/WASM build for the Chromebook.

Model:
onnx-community/SmolLM2-360M-Instruct-ONNX

Quantization:
Q4F16

Runtime:
Transformers.js

The model is loaded with:
- device: wasm
- browser cache enabled
- WASM cache enabled

The model repo currently provides a 272 MB `model_q4f16.onnx` file and a
Transformers.js usage path. The browser caches model files after the first
download, so later visits do not need to download them again.

## GitHub Pages

Upload:
- index.html
- app.js
- sw.js
- manifest.json
- README.md

Enable GitHub Pages and open the HTTPS Pages URL.

Do not use `file://`.

## Important

This version intentionally uses CPU/WASM because the Chromebook's WebGPU
path was losing its GPU device during Qwen 0.5B initialization.

CPU/WASM is slower, but it avoids WebGPU entirely.

The app uses a Web Worker only if you later add one; this current build keeps
the runtime simple and compatible first.

## UI

The dashboard shows:
- CPU/WASM backend
- JS heap, when Chromium exposes it
- browser origin storage usage/quota
- output tokens and tok/s
- persistent estimated equivalent spend

The cost numbers are only configurable estimates, not real provider bills.

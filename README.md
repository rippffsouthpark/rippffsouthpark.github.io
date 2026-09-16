# Qwen 2.5 0.5B — CPU/WASM browser build

This version deliberately does NOT use WebGPU. It uses Transformers.js and the ONNX Community Qwen2.5-0.5B-Instruct model with ONNX Runtime Web/WASM.

Model repository:
`onnx-community/Qwen2.5-0.5B-Instruct`

The model repository is explicitly structured for Transformers.js. Its main ONNX model is approximately 1.99 GB, so this CPU build is much larger than the 266 MB WebLLM cache you were using before.

## GitHub Pages

Upload:
- index.html
- app.js
- sw.js
- manifest.json
- README.md

Enable GitHub Pages and open the HTTPS URL.

Do not open index.html directly with file://.

## Caching

Transformers.js caches model artifacts in browser storage, and the service worker caches the small application shell.

The model is downloaded once per browser/origin when not already cached.

## Important

CPU/WASM avoids the Chromebook's WebGPU device-loss failure, but it is expected to be significantly slower than a healthy WebGPU path.

The dashboard's cost figures are user-editable estimates, not actual vendor accounting.

The all-time estimated spend is kept in localStorage.

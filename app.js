import * as webllm from "https://esm.run/@mlc-ai/web-llm";

const model = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";

const appConfig = {
  ...webllm.prebuiltAppConfig,
  cacheBackend: "indexeddb",
};

const engine = await webllm.CreateMLCEngine(
  model,
  {
    appConfig,
    initProgressCallback: (p) => console.log(p.text),
  },
  {
    context_window_size: 2048,
  }
);
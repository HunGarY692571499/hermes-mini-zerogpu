---
title: Mini Local Chat ZeroGPU
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
startup_duration_timeout: 30m
---

# Mini-LLM auf ZeroGPU

Kein Hermes-Agent. Request-GPU via `@spaces.GPU(duration=90)`.

## Fix 2026-09-28
Live-Space war `RUNTIME_ERROR` / `Exit code: 0`:
- `app.py` beendete ohne `demo.launch()`
- Modell-Load lief lazy + `device_map="auto"` außerhalb des GPU-Workers
- `requirements.txt` listete `gradio`, `spaces`, unpinned `torch` (ZeroGPU-Drift)

Default-Modell: `Qwen/Qwen2.5-1.5B-Instruct`  
Override: Space-Variable `MODEL_ID` oder `LOCAL_MODEL_ID`.

## UI
- editierbarer Systemprompt
- Frameworks: CoT, Plan-then-Solve, ReAct-Lite, Self-Refine, ToT-Lite, Socratic, First Principles, Devil's Advocate, Expert Panel
- temperature / max_new_tokens

## API
- `GET /v1/models`
- `POST /v1/chat/completions`

Hardware im Space-Settings: **ZeroGPU**. Pause aus.

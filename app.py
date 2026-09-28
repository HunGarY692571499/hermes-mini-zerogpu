"""ZeroGPU Mini-LLM + OpenAI-compatible /v1.

Kill-bugs this revision closes:
1. Missing demo.launch() → Space exits 0 → RUNTIME_ERROR
2. Model load outside @spaces.GPU via device_map=auto → CUDA in web process
3. requirements pinned gradio/spaces/unpinned torch → ZeroGPU drift
"""
from __future__ import annotations

import spaces  # MUST be first among CUDA-touching imports

import json
import os
import time
import uuid
from typing import Any

import gradio as gr
import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID") or os.environ.get(
    "LOCAL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"
)
MAX_NEW_DEFAULT = int(os.environ.get("LOCAL_MAX_NEW", "256"))

FRAMEWORKS = {
    "none": "",
    "CoT": "Think step by step. Put reasoning first, then the final answer.",
    "Plan-then-Solve": "First write a short plan. Then execute the plan. Then answer.",
    "ReAct-Lite": "Alternate Thought / Action / Observation. Stop when you can answer.",
    "Self-Refine": "Draft an answer, critique it, then output a refined final answer.",
    "ToT-Lite": "List 2-3 candidate paths, pick the strongest, then answer.",
    "Socratic": "Ask the key questions first, then answer from those questions.",
    "First Principles": "Strip assumptions. Rebuild from primitives. Then answer.",
    "Devil's Advocate": "State the strongest counter-argument, then the best synthesis.",
    "Expert Panel": "Give 2 expert views, then a chair synthesis as the answer.",
}

# Module-scope load: ZeroGPU intercepts .to("cuda") and packs weights.
# Do NOT use device_map="auto" and do NOT lazy-load from a non-GPU helper.
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.to("cuda")
model.eval()
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return " ".join(parts)
    return str(content)


def _messages_to_prompt(messages: list[dict], tools: list | None = None) -> str:
    sys_extra = ""
    if tools:
        sys_extra = (
            "\nYou may call tools. To call, output ONLY:\n"
            '<tool_call>{"name":"TOOL","arguments":{...}}</tool_call>\n'
            f"Tools: {json.dumps(tools)[:4000]}\n"
        )
    conv: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        content = _as_text(m.get("content"))
        if role == "system":
            conv.append({"role": "system", "content": content + sys_extra})
            sys_extra = ""
        elif role == "tool":
            conv.append({"role": "user", "content": f"Tool result: {content}"})
        else:
            conv.append({"role": role, "content": content})
    if sys_extra:
        conv.insert(0, {"role": "system", "content": sys_extra.strip()})
    return tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


def _parse_tools(text: str) -> tuple[str, list[dict]]:
    calls: list[dict] = []
    out = text
    start = text.find("<tool_call>")
    end = text.find("</tool_call>")
    if start != -1 and end != -1 and end > start:
        raw = text[start + len("<tool_call>") : end].strip()
        try:
            obj = json.loads(raw)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": obj.get("name", "unknown"),
                        "arguments": json.dumps(obj.get("arguments", {})),
                    },
                }
            )
            out = (text[:start] + text[end + len("</tool_call>") :]).strip()
        except Exception:
            pass
    return out, calls


@spaces.GPU(duration=90)
def generate_text(
    prompt: str,
    max_new_tokens: int = MAX_NEW_DEFAULT,
    temperature: float = 0.7,
) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    temp = max(float(temperature), 0.01)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=temp,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


def complete(body: dict) -> dict:
    messages = body.get("messages") or [
        {"role": "user", "content": body.get("prompt", "")}
    ]
    tools = body.get("tools")
    prompt = _messages_to_prompt(messages, tools)
    max_new = int(body.get("max_tokens") or MAX_NEW_DEFAULT)
    temp = float(body.get("temperature") or 0.7)
    text = generate_text(prompt, max_new, temp)
    content, calls = _parse_tools(text)
    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def ui_chat(message, history, system_prompt, framework, temperature, max_new_tokens):
    if isinstance(message, dict):
        message = _as_text(message.get("content") or message.get("text") or message)
    hist: list[dict] = []
    extra = FRAMEWORKS.get(framework or "none", "")
    sys_txt = (system_prompt or "").strip()
    if extra:
        sys_txt = (sys_txt + "\n" + extra).strip()
    if sys_txt:
        hist.append({"role": "system", "content": sys_txt})
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            if role in {"user", "assistant", "system"}:
                hist.append({"role": role, "content": _as_text(turn.get("content"))})
        else:
            u, a = turn
            hist.append({"role": "user", "content": u})
            hist.append({"role": "assistant", "content": a})
    hist.append({"role": "user", "content": str(message)})
    r = complete(
        {
            "messages": hist,
            "temperature": temperature,
            "max_tokens": int(max_new_tokens),
        }
    )
    return r["choices"][0]["message"].get("content") or "(tool call)"


with gr.Blocks(title="hermes-mini-zerogpu") as demo:
    gr.Markdown(
        f"# Mini LLM ZeroGPU\n`{MODEL_ID}` — Request-GPU. Kein Hermes-Gateway."
    )
    with gr.Row():
        system_prompt = gr.Textbox(
            label="Systemprompt",
            value="You are a concise technical assistant.",
            lines=3,
        )
        framework = gr.Dropdown(
            choices=list(FRAMEWORKS.keys()),
            value="none",
            label="Reasoning-Framework",
        )
    with gr.Row():
        temperature = gr.Slider(0.01, 1.5, value=0.7, step=0.05, label="temperature")
        max_new_tokens = gr.Slider(
            16, 512, value=MAX_NEW_DEFAULT, step=16, label="max_new_tokens"
        )
    gr.ChatInterface(
        fn=ui_chat,
        additional_inputs=[system_prompt, framework, temperature, max_new_tokens],
        type="messages",
        fill_height=True,
    )


@demo.app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local-zerogpu"}],
    }


@demo.app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        return complete(body)
    except Exception as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "zerogpu_error"}},
            status_code=503,
        )


if __name__ == "__main__":
    demo.queue()
    demo.launch(ssr_mode=False, show_error=True)

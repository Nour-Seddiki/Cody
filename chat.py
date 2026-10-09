"""
Chat with the base model, or with a fine-tuned LoRA adapter on top of it.

    .\\run.ps1 chat.py                                  # base Qwen3.5-4B
    $env:ADAPTER="checkpoints/adapter"; .\\run.ps1 chat.py
    $env:PROMPT="Write a Python function that reverses a linked list"; .\\run.ps1 chat.py   # one-shot
"""
import os

import torch
from transformers import TextStreamer

from model import MODEL_ID, load_tokenizer, load_with_adapter
from runtime import configure_runtime

ADAPTER = os.environ.get("ADAPTER")
PROMPT = os.environ.get("PROMPT")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
SYSTEM = os.environ.get("SYSTEM", "You are Cody, a concise and precise coding assistant.")

configure_runtime("cuda")
tokenizer = load_tokenizer()
print(f"loading {MODEL_ID}" + (f" + adapter {ADAPTER}" if ADAPTER else ""))
model = load_with_adapter(ADAPTER, quantize=os.environ.get("QUANTIZE", "1") == "1").eval()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


def reply(history):
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7,
                             top_p=0.8, top_k=20, streamer=streamer)
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


history = [{"role": "system", "content": SYSTEM}]
if PROMPT:
    reply(history + [{"role": "user", "content": PROMPT}])
else:
    print("type a message (empty line or Ctrl+C to quit)")
    while (msg := input("\nyou> ").strip()):
        history.append({"role": "user", "content": msg})
        print("cody> ", end="", flush=True)
        history.append({"role": "assistant", "content": reply(history)})

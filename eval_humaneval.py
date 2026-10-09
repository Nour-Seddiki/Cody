"""
HumanEval pass@1 (greedy) for the base model or base + LoRA adapter - the before/after check for a
fine-tune. 164 Python problems; each answer is run against the problem's hidden unit tests.

    .\\run.ps1 eval_humaneval.py                                   # base model
    $env:ADAPTER="checkpoints/adapter"; .\\run.ps1 eval_humaneval.py
    $env:N_PROBLEMS=20; .\\run.ps1 eval_humaneval.py               # quick subset

The model is asked in chat form (same system prompt as chat.py / the training data) to return the
complete function in a ```python block. The generated code is executed in a subprocess with a
timeout, as in the reference HumanEval harness - it is model-written code, run inside WSL.
Results go to results/humaneval_<base|adapter name>.json.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from model import MODEL_ID, load_tokenizer, load_with_adapter
from runtime import configure_runtime

ADAPTER = os.environ.get("ADAPTER")
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", 164))
GEN_BATCH = int(os.environ.get("GEN_BATCH", 8))  # not BATCH_SIZE: that one is train.py's
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 768))
SYSTEM = os.environ.get("SYSTEM", "You are Cody, a concise and precise coding assistant.")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

path = hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
                       repo_type="dataset")
problems = pq.read_table(path).to_pylist()[:N_PROBLEMS]
TASKS = os.environ.get("TASKS")  # e.g. "HumanEval/27,HumanEval/161" - rerun just these
if TASKS:
    problems = [p for p in problems if p["task_id"] in TASKS.split(",")]

configure_runtime("cuda")
tokenizer = load_tokenizer()
tokenizer.padding_side = "left"  # generation: prompts end flush at the right edge
print(f"loading {MODEL_ID}" + (f" + adapter {ADAPTER}" if ADAPTER else ""))
model = load_with_adapter(ADAPTER).eval()


def chat_prompt(problem):
    ask = ("Complete the following Python function. Return the complete function, including the "
           f"signature and any imports it needs, in a single ```python code block.\n\n```python\n{problem['prompt']}```")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def extract_code(text, problem):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    code = max(blocks, key=len) if blocks else text
    if f"def {problem['entry_point']}" not in code:  # body only: graft it onto the given signature
        code = problem["prompt"] + code
    # keep the prompt's imports (typing etc.) - answers often drop them
    imports = "\n".join(l for l in problem["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return imports + "\n" + code


def passes(program, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        with open(f, "w") as fh:
            fh.write(program)
        try:
            r = subprocess.run([sys.executable, f], cwd=d, capture_output=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


# longest prompts first, batched by similar length to keep padding low
order = sorted(range(len(problems)), key=lambda i: -len(problems[i]["prompt"]))
results, t0, n_tokens = {}, time.time(), 0
for b in range(0, len(order), GEN_BATCH):
    idx = order[b:b + GEN_BATCH]
    enc = tokenizer([chat_prompt(problems[i]) for i in idx], return_tensors="pt", padding=True,
                    add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    for row, i in enumerate(idx):
        gen = out[row, enc["input_ids"].shape[1]:]
        n_tokens += int((gen != tokenizer.pad_token_id).sum())
        text = tokenizer.decode(gen, skip_special_tokens=True)
        p = problems[i]
        program = extract_code(text, p) + "\n\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
        results[p["task_id"]] = {"passed": passes(program), "completion": text}
    done = len(results)
    n_pass = sum(r["passed"] for r in results.values())
    print(f"{done}/{len(problems)} | pass@1 so far {n_pass / done:.3f} | "
          f"{n_tokens / (time.time() - t0):.0f} gen tok/s", flush=True)

n_pass = sum(r["passed"] for r in results.values())
name = (os.path.basename(os.path.normpath(ADAPTER)) if ADAPTER else "base") + (f"_b{GEN_BATCH}_subset" if TASKS else "")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"humaneval_{name}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"model": MODEL_ID, "adapter": ADAPTER, "n": len(problems), "pass@1": n_pass / len(problems),
               "results": results}, f, indent=1)
print(f"HumanEval pass@1 = {n_pass}/{len(problems)} = {n_pass / len(problems):.3f} -> {out_path}")

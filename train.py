"""
QLoRA supervised fine-tuning of Qwen3.5-4B on chat-format JSONL. Fits the 8 GB RTX 5050: the base
model is frozen in 4-bit NF4 (~3 GB), only the LoRA adapters (~1% of the weights) are trained, with
gradient checkpointing.

Data: one JSON object per line, {"messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...]}.
The loss is on the final assistant message only; everything before it is context.

Usage (from the repo root, via the launcher that runs it in WSL):
    .\\run.ps1 train.py                                       # data/sample.jsonl, defaults below
    $env:TRAIN_FILE="data/my_code.jsonl"; .\\run.ps1 train.py
    $env:MAX_STEPS=5; $env:EVAL_EVERY=5; .\\run.ps1 train.py  # smoke test
"""
import json
import math
import os
import random
import time

import torch

from model import MODEL_ID, add_lora, load_base, load_tokenizer
from runtime import configure_runtime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# config - env-var overridable
TRAIN_FILE = os.environ.get("TRAIN_FILE", os.path.join(SCRIPT_DIR, "data", "sample.jsonl"))
VAL_FILE = os.environ.get("VAL_FILE")  # if unset, VAL_FRACTION of TRAIN_FILE is held out
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", 0.05))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(SCRIPT_DIR, "checkpoints", "adapter"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 1024))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
EPOCHS = int(os.environ.get("EPOCHS", 2))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))  # optimizer steps; 0 = run all epochs
LR = float(os.environ.get("LR", 2e-4))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.03))
LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 32))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", 0.05))
QUANTIZE = os.environ.get("QUANTIZE", "1") == "1"
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 100))
LOG_EVERY = int(os.environ.get("LOG_EVERY", 5))
SEED = 1337

random.seed(SEED)
torch.manual_seed(SEED)
assert torch.cuda.is_available(), "a CUDA GPU is required"
configure_runtime("cuda")

# ----------------------------------------------------------------------------
# data
tokenizer = load_tokenizer()


def encode(messages):
    """Token ids + labels with everything except the final assistant reply masked to -100.
    Prompt and reply are tokenized separately so the mask boundary is exact - the same split that
    happens at inference, where the prompt is tokenized and the reply generated."""
    assert messages[-1]["role"] == "assistant", "the last message must be the assistant reply"
    # enable_thinking=False: train the direct-answer mode (empty <think></think> block)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    assert full.startswith(prompt), "chat template rendered the prompt differently in the full chat"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + reply_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(prompt_ids) + reply_ids)[:MAX_SEQ_LEN]
    return ids, labels


def load_jsonl(path):
    examples, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids, labels = encode(json.loads(line)["messages"])
                if any(l != -100 for l in labels):
                    examples.append((ids, labels))
                else:
                    dropped += 1  # the prompt alone filled MAX_SEQ_LEN
    if dropped:
        print(f"{path}: dropped {dropped} examples whose reply was truncated away (MAX_SEQ_LEN={MAX_SEQ_LEN})")
    return examples


train_data = load_jsonl(TRAIN_FILE)
if VAL_FILE:
    val_data = load_jsonl(VAL_FILE)
else:
    random.shuffle(train_data)
    n_val = max(1, int(len(train_data) * VAL_FRACTION))
    val_data, train_data = train_data[:n_val], train_data[n_val:]
print(f"train examples: {len(train_data)}, val examples: {len(val_data)}")


def batches(data, shuffle):
    order = list(range(len(data)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), BATCH_SIZE):
        chunk = [data[j] for j in order[i:i + BATCH_SIZE]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), tokenizer.pad_token_id)
        labels = torch.full((len(chunk), width), -100)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (x, y) in enumerate(chunk):  # right padding
            ids[row, :len(x)], labels[row, :len(y)], mask[row, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids.cuda(), labels.cuda(), mask.cuda()


# ----------------------------------------------------------------------------
# model
print(f"loading {MODEL_ID} (4-bit={QUANTIZE})")
model = add_lora(load_base(QUANTIZE), r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
model.print_trainable_parameters()
print(f"VRAM after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps_per_epoch = math.ceil(math.ceil(len(train_data) / BATCH_SIZE) / GRAD_ACCUM_STEPS)
total_steps = MAX_STEPS or steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))


def lr_at(step):  # linear warmup, then cosine to 10% of LR
    if step < warmup_steps:
        return LR * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def reply_loss(ids, labels, mask):
    """Mean cross-entropy over the reply tokens, and how many there are. The lm_head runs only on
    the positions that predict a reply token (logits_to_keep): with Qwen3.5's 248k vocabulary,
    full-sequence logits are ~1 GB per 1k tokens in fp32 and most of them would be masked anyway."""
    keep = (labels[:, 1:] != -100).any(0).nonzero().squeeze(1)  # position p predicts token p + 1
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
    targets = labels[:, keep + 1]
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
    return loss, (targets != -100).sum().item()


@torch.no_grad()
def evaluate():
    model.eval()
    total, n_tokens = 0.0, 0
    for ids, labels, mask in batches(val_data, shuffle=False):
        loss, n = reply_loss(ids, labels, mask)
        total += loss.item() * n
        n_tokens += n
    model.train()
    return total / max(1, n_tokens)


# ----------------------------------------------------------------------------
# train
print(f"training for {total_steps} optimizer steps "
      f"(batch {BATCH_SIZE} x grad accum {GRAD_ACCUM_STEPS}, {steps_per_epoch} steps/epoch)")
model.train()
step, micro, best_val = 0, 0, float("inf")
t0, tokens_since_log, loss_since_log, steps_since_log = time.time(), 0, 0.0, 0
done = False
for epoch in range(EPOCHS if not MAX_STEPS else 10 ** 9):
    for ids, labels, mask in batches(train_data, shuffle=True):
        loss, _ = reply_loss(ids, labels, mask)
        (loss / GRAD_ACCUM_STEPS).backward()
        loss_since_log += loss.item() / GRAD_ACCUM_STEPS
        tokens_since_log += mask.sum().item()
        micro += 1
        if micro % GRAD_ACCUM_STEPS:
            continue

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        steps_since_log += 1

        if step % LOG_EVERY == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step}/{total_steps} | epoch {epoch} | loss {loss_since_log / steps_since_log:.4f} | "
                  f"lr {lr_at(step - 1):.2e} | {tokens_since_log / dt:.0f} tok/s | "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            t0, tokens_since_log, loss_since_log, steps_since_log = time.time(), 0, 0.0, 0

        if step % EVAL_EVERY == 0 or step == total_steps:
            t_eval = time.time()
            val_loss = evaluate()
            print(f"step {step} | val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.2f}")
            if val_loss < best_val:
                best_val = val_loss
                model.save_pretrained(OUT_DIR)
                tokenizer.save_pretrained(OUT_DIR)
                print(f"saved best adapter -> {OUT_DIR}")
            t0 += time.time() - t_eval  # don't count eval (or saving) time in tok/s

        if step >= total_steps:
            done = True
            break
    if done:
        break

print(f"done. best val loss {best_val:.4f}, adapter at {OUT_DIR}")

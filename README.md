# Cody

QLoRA fine-tuning of **Qwen3.5-4B** into a Python coding assistant on an 8 GB laptop GPU
(RTX 5050 Laptop), with a before/after HumanEval evaluation.

## Results

Fine-tuned for 1 epoch on 3,000 verified examples from
[OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct). HumanEval is 164
problems, pass@1, greedy decoding, model loaded in 4-bit, same prompt and settings for both rows:

| model | HumanEval pass@1 | 
|---|---|
| Qwen3.5-4B (base) | **125 / 164 = 76.2%** |
| + Cody LoRA (OpenCodeInstruct, 3k examples) | 116 / 164 = 70.7% |

**The fine-tune did not make the model better at coding.** Per problem, it fixed 12 and broke 21.
An exact McNemar test gives p = 0.16, so the drop isn't statistically significant, but there is
clearly no gain. What changed:

- **Fixed:** all 5 base failures where greedy decoding got stuck in a repetition loop (e.g.
  `''.join([''.join([...` on `flip_case`). The fine-tuned model ends its answers cleanly.
- **Broke:** 21 problems now fail on plain logic mistakes. The base model often reasoned in code
  comments before committing to an answer. After training on short, code-first answers, it
  answers straight away: replies are 10% shorter (666 vs 743 characters), and it gets more
  tricky problems wrong.
- **Why:** the training answers come from Qwen2.5-Coder-32B (2024). Qwen3.5-4B (2026) is already
  about as good, so imitating that data mostly teaches it the style, not new skills. Validation
  loss still dropped (0.14 → 0.126), because the model got better at predicting *that
  dataset's* answers. Lower loss on the data doesn't mean better code.

Ideas that are more likely to help: training on data from a stronger teacher or with
reasoning traces (thinking mode); keeping the model's own correct answers on similar problems
(rejection sampling / self-distillation); or narrowing the goal to something the base model
doesn't know, such as your own codebase or a niche library. Per-problem results, completions
included, are in [`results/`](results/).

Training: 375 steps (batch 1 × 8 accumulation), LR 1e-4, LoRA r=16, about 50 min at ~460
tokens/s, 5.5 GiB peak VRAM. The 122 MB adapter isn't in this repo because GitHub caps files at
100 MB. Re-create it with the commands below.

## Why this model

| | |
|---|---|
| Model | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), released 2026-02. Instruct/thinking model, 262k context, Apache-2.0 |
| Why it fits | Loaded in 4-bit NF4, the frozen base takes ~3 GB of VRAM. Only LoRA adapters are trained (r=16, 30M parameters, 0.7%), with gradient checkpointing |
| Bigger options | Qwen3.5-9B doesn't leave room for training activations in 8 GB. Qwen3.6 / 3.8 are 27B+ only |
| Smaller fallback | `MODEL_ID=Qwen/Qwen3.5-2B` gives faster steps and room for longer sequences |

Qwen3.5 checkpoints are vision-language models. `AutoModelForCausalLM` loads only the text
weights, so the vision tower costs nothing here. 3 of every 4 layers are Gated DeltaNet
(linear attention). Their fast Triton kernels come from `flash-linear-attention`, which only runs on Linux.

Two memory tricks keep training inside 8 GB:

- **No `prepare_model_for_kbit_training`:** it upcasts every non-quantized weight to fp32,
  including the 248k × 2560 embedding table, which costs 1.2 GiB for nothing. `model.py` turns on
  gradient checkpointing and input gradients directly instead.
- **Reply-only logits:** `train.py` passes `logits_to_keep` so the output layer runs only on the
  positions whose next token is part of the reply. With a 248k-word vocabulary, full-sequence
  logits are about 1 GB per 1k tokens in fp32, and most of them would be masked out anyway. This
  is what made batch 2 run out of memory.

## Layout

| file | |
|---|---|
| `model.py` | load base (4-bit) / attach or load LoRA |
| `train.py` | the training loop: reply-only loss, warmup + cosine LR, best-val adapter saving |
| `prepare_data.py` | OpenCodeInstruct shard → filtered `data/oci_{train,val}.jsonl` |
| `eval_humaneval.py` | HumanEval pass@1, batched greedy generation, answers executed in a subprocess |
| `chat.py` | interactive or one-shot chat, base or with an adapter |
| `download_model.py` | resumable, SHA-256-checked weight download (works on a flaky connection) |
| `run.ps1` | Windows launcher: runs a script in WSL, keeps the machine awake, forwards env vars |
| `setup_wsl.sh` | one-time WSL env: uv, torch cu128, transformers, peft, bitsandbytes, fla |
| `runtime.py` | CUDA allocator cap (the Windows driver spills into shared RAM instead of OOMing) |
| `data/` | `sample.jsonl` (20 examples, for a smoke test) and the prepared OpenCodeInstruct split |
| `results/` | HumanEval per-problem results for the base model and the adapter |

## Setup

The scripts run inside **WSL Ubuntu**. On the machine this was built on, Windows Smart App
Control blocks torch's unsigned `torch_python.dll`, so torch doesn't import on Windows. Linux also
has the Triton kernels Qwen3.5 needs. From the repo root in PowerShell:

```powershell
wsl -d Ubuntu -- bash setup_wsl.sh
```

Download the weights (~9.3 GB) on Windows, into `%USERPROFILE%\models\Qwen3.5-4B`. Keep them out
of OneDrive-synced folders. This step needs only `huggingface_hub` and `httpx`, not torch:

```powershell
python download_model.py
```

Optionally copy them onto the WSL disk. `run.ps1` prefers that copy, which loads in seconds
instead of ~2.5 min through `/mnt/c`:

```powershell
wsl -d Ubuntu --exec bash -c 'mkdir -p ~/models && cp -r /mnt/c/Users/$USER/models/Qwen3.5-4B ~/models/'
```

(If your Linux user name differs from your Windows one, use the Windows path directly.)

## Use

```powershell
.\run.ps1 chat.py                                                     # base model
$env:MAX_STEPS=5; $env:EVAL_EVERY=5; .\run.ps1 train.py               # smoke test on data/sample.jsonl
Remove-Item env:MAX_STEPS, env:EVAL_EVERY

# the run from Results
$env:TRAIN_FILE="data/oci_train.jsonl"; $env:VAL_FILE="data/oci_val.jsonl"; $env:EPOCHS=1
$env:LR="1e-4"; $env:EVAL_EVERY=75; $env:OUT_DIR="checkpoints/oci_adapter"
.\run.ps1 train.py

.\run.ps1 eval_humaneval.py                                           # base
$env:ADAPTER="checkpoints/oci_adapter"; .\run.ps1 eval_humaneval.py   # fine-tuned
$env:ADAPTER="checkpoints/oci_adapter"; .\run.ps1 chat.py
```

### Data

JSONL, one conversation per line. The loss is on the **last assistant message** only:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`data/oci_*.jsonl` comes from [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
(© NVIDIA, CC-BY-4.0): Python Q&As answered by Qwen2.5-Coder-32B-Instruct, each with
unit tests, their pass rate, and an LLM-judge score. `prepare_data.py` takes one shard (100k rows)
and keeps answers that pass **all** their unit tests, score ≥ 4/5 from the judge on every criterion,
and fit in 1,024 tokens. Longer ones are dropped, not truncated. Of the 23k that qualify, it samples
3,000 train and 150 val examples (440 tokens on average):

```powershell
.\run.ps1 prepare_data.py /mnt/c/path/to/train-00000-of-00050.parquet
```

### Main knobs (env vars)

| var | default | |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3.5-4B` (`run.ps1` points it at the local copy) | |
| `MAX_SEQ_LEN` | 1024 | |
| `BATCH_SIZE` × `GRAD_ACCUM_STEPS` | 1 × 8 | batch 2 runs out of memory on long pairs |
| `EPOCHS` / `MAX_STEPS` | 2 / 0 | `MAX_STEPS` > 0 overrides epochs |
| `LR` | 2e-4 | warmup + cosine to 10% |
| `LORA_R` / `LORA_ALPHA` | 16 / 32 | |
| `CUDA_MEM_FRACTION` | 0.85 (set by `run.ps1`) | caps the CUDA cache below VRAM |
| `N_PROBLEMS` / `TASKS` / `GEN_BATCH` | 164 / all / 8 | HumanEval subset, specific task ids, generation batch |

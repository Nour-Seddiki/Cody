# Cody

QLoRA fine-tuning of **Qwen3.5-4B** into a Python coding assistant on an 8 GB laptop GPU
(RTX 5050 Laptop), with a before/after HumanEval evaluation.

## Results

Two LoRA fine-tunes of Qwen3.5-4B, compared on HumanEval: 164 problems, pass@1, greedy
decoding, model loaded in 4-bit, same prompt and settings for every row.

| model | training data | HumanEval pass@1 | vs base (fixed / broken) |
|---|---|---|---|
| Qwen3.5-4B (base) | none | 125 / 164 = 76.2% | |
| + OpenCodeInstruct LoRA | 3,000 answers by Qwen2.5-Coder-32B | 116 / 164 = 70.7% | +12 / −21 (p = 0.16) |
| + self-distilled LoRA | 1,163 of the model's own answers that pass their unit tests | **126 / 164 = 76.8%** | +8 / −7 (p = 1.0) |

p values are exact McNemar tests on the per-problem results. **Neither fine-tune improves on
the base model in a way HumanEval can detect.** Training on another model's answers made it
worse. Training on its own verified answers kept its quality and moved the score by one
problem, which is noise. Per-problem results, completions included, are in [`results/`](results/).

### 1. OpenCodeInstruct: imitating another model's answers hurt

1 epoch on 3,000 verified examples from
[OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct): every answer passes all
its unit tests and scored ≥ 4/5 from the LLM judge.

- **Fixed:** all 5 base failures where greedy decoding got stuck in a repetition loop (e.g.
  `''.join([''.join([...` on `flip_case`).
- **Broke:** 21 problems that now fail on plain logic mistakes. The base model often reasoned in
  code comments before committing to an answer. After training on short, code-first answers, it
  answers straight away: replies got 10% shorter (666 vs 743 characters).
- **Why:** the answers come from Qwen2.5-Coder-32B (2024), and Qwen3.5-4B (2026) is already about
  as good. Imitating them mostly taught the style, not new skills. Validation loss still fell
  (0.14 → 0.126), because the model got better at predicting *that dataset*. Lower loss on the
  data doesn't mean better code.

### 2. Self-distillation (rejection sampling): no regression, no measurable gain

`selfdistill.py` has the model answer 800 fresh OpenCodeInstruct questions (none used in run 1),
4 sampled answers each, in the same non-thinking mode the eval uses. Every answer runs against
the question's unit tests in a sandbox, and only passing answers are kept: 2,279 of 3,200 (71%).
Training uses up to 2 distinct passing answers per question.

- **Kept its quality:** answer length (722 vs 743 characters) and behaviour stay close to the
  base model's, as intended. The per-problem changes cancel out: 8 fixed, 7 broken.
- **Why so little gain:** most of the signal was already known to the model. 473 of the 800
  questions were solved by all 4 samples, so most training examples repeat what it already does.
  Validation loss was 0.069 from the start, and the model already predicts its own answers
  almost perfectly. The second epoch only memorised (val 0.0687 → 0.0695), so the epoch-1
  checkpoint is the one evaluated.
- **Repetition loops remain:** the 5 greedy-decoding loops are still there. The filter only ever
  saw sampled answers, which rarely loop.

Fixing a harness bug mattered here: many questions never name the function, but their tests
call the reference solution's name (`longest_palindrome` where the model wrote
`longest_palindromic_substring`). 85% of the first "failures" were that `NameError`, not wrong
code. The test runner now aliases each missing name to the answer's closest-named function. The
answer still has to pass the tests, but it's no longer rejected for a naming choice the
question never specified.

**What would likely help next:** sample only the questions the model gets *sometimes* right
(1–3 of 4 here) and many more of them, so training focuses on what it can learn; use thinking
mode, whose reasoning traces carry more signal; or narrow the goal to something the base model
doesn't know, such as your own codebase or a niche library.

### Training runs

| | OpenCodeInstruct | self-distilled |
|---|---|---|
| examples (train / val) | 3,000 / 150 | 1,163 / 58 (val split by question) |
| schedule | 1 epoch, 375 steps | 2 epochs, 292 steps (best = end of epoch 1) |
| common | batch 1 × 8 accumulation, LR 1e-4 cosine, LoRA r=16 | |
| time / speed / VRAM | ~50 min, ~460 tok/s, 5.5 GiB | ~35 min, ~430 tok/s, 5.4 GiB |
| sampling | | 800 questions × 4 in ~85 min at batch 24 (12 for the longest prompts) |

The adapters (122 MB each) aren't in this repo because GitHub caps files at 100 MB. Re-create
them with the commands below.

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
| `selfdistill.py` | self-distillation: sample the model's own answers, run their unit tests in a sandbox, keep the passing ones → `data/sd_{train,val}.jsonl` |
| `eval_humaneval.py` | HumanEval pass@1, batched greedy generation, answers executed in a subprocess |
| `chat.py` | interactive or one-shot chat, base or with an adapter |
| `download_model.py` | resumable, SHA-256-checked weight download (works on a flaky connection) |
| `run.ps1` | Windows launcher: runs a script in WSL, keeps the machine awake, forwards env vars |
| `setup_wsl.sh` | one-time WSL env: uv, torch cu128, transformers, peft, bitsandbytes, fla |
| `runtime.py` | CUDA allocator cap (the Windows driver spills into shared RAM instead of OOMing) |
| `data/` | `sample.jsonl` (20 examples, for a smoke test), the OpenCodeInstruct split, the self-distilled split |
| `results/` | HumanEval per-problem results for the base model and both adapters |

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

# run 1 from Results: OpenCodeInstruct
$env:TRAIN_FILE="data/oci_train.jsonl"; $env:VAL_FILE="data/oci_val.jsonl"; $env:EPOCHS=1
$env:LR="1e-4"; $env:EVAL_EVERY=75; $env:OUT_DIR="checkpoints/oci_adapter"
.\run.ps1 train.py

# run 2: self-distillation (sampling resumes if stopped; use GEN_BATCH=12 if the long prompts run out of memory)
$env:N_QUESTIONS=800; $env:K=4; $env:GEN_BATCH=24
.\run.ps1 selfdistill.py /mnt/c/path/to/train-00000-of-00050.parquet
$env:TRAIN_FILE="data/sd_train.jsonl"; $env:VAL_FILE="data/sd_val.jsonl"; $env:EPOCHS=2
$env:LR="1e-4"; $env:EVAL_EVERY=73; $env:OUT_DIR="checkpoints/sd_adapter"
.\run.ps1 train.py

.\run.ps1 eval_humaneval.py                                           # base
$env:ADAPTER="checkpoints/sd_adapter"; .\run.ps1 eval_humaneval.py    # fine-tuned
$env:ADAPTER="checkpoints/sd_adapter"; .\run.ps1 chat.py
```

`selfdistill.py` runs model-written code. Each test runs in fresh mount, network and PID
namespaces, with the Windows drives unmounted, no network, as user `nobody`, under a 2 GB memory
limit and a 10 s timeout. `RESCORE=1` re-runs the tests on the saved samples without generating,
e.g. after a harness fix.

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

`data/sd_*.jsonl` holds 1,163 / 58 answers written by Qwen3.5-4B itself (Apache-2.0) to 617 other
questions from the same shard. Each answer passes all of that question's unit tests.

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
| `N_QUESTIONS` / `K` / `KEEP_PER_QUESTION` | 1500 / 4 / 2 | self-distillation: questions, samples per question, passing answers kept per question |

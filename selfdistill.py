"""
Self-distillation data by rejection sampling: Qwen3.5-4B answers OpenCodeInstruct questions itself,
K sampled answers per question, and every answer is run against the question's unit tests. Answers
that pass all of them become the SFT set - the model's own style (including its habit of reasoning
in comments) with only the correct outcomes kept, instead of imitating another model's answers
(which cost 5.5 HumanEval points with the plain OpenCodeInstruct fine-tune - see README).

    .\\run.ps1 selfdistill.py <shard.parquet>     # N_QUESTIONS=1500 K=4 by default

Questions: the same verified pool as prepare_data.py (reference answer passes all tests, judge
scores >= 4), minus the ones already used in data/oci_*.jsonl. Raw samples are appended to
checkpoints/selfdistill_raw.jsonl as they finish, so an interrupted run resumes where it stopped.
Output: data/sd_train.jsonl, data/sd_val.jsonl.

The answers are model-written code. Each test runs in a sandbox: fresh mount/network/pid
namespaces with the Windows drives (/mnt/c, ...) unmounted, no network, as user nobody, with a
2 GB address-space limit and a timeout.
"""
import ast
import builtins
import difflib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import torch

from model import MODEL_ID, load_tokenizer, load_with_adapter
from runtime import configure_runtime

SHARD = sys.argv[1]
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", 1500))
K = int(os.environ.get("K", 4))  # samples per question
GEN_BATCH = int(os.environ.get("GEN_BATCH", 32))  # sequences per generate() call (questions x K)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 768))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 1024))
KEEP_PER_QUESTION = int(os.environ.get("KEEP_PER_QUESTION", 2))
VAL_FRACTION = 0.05
RESCORE = os.environ.get("RESCORE") == "1"  # skip generation, re-test the saved samples
SYSTEM = "You are Cody, a concise and precise coding assistant."  # same as prepare_data.py / chat.py
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "checkpoints", "selfdistill_raw.jsonl")

# ----------------------------------------------------------------------------
# questions
rows = pq.read_table(SHARD, columns=["id", "input", "unit_tests", "llm_judgement", "average_test_score"]).to_pylist()


def judged_ok(raw):
    try:
        return all(v["score"] >= 4 for v in json.loads(raw).values())
    except (ValueError, KeyError, TypeError):
        return False


used = set()
for name in ["oci_train.jsonl", "oci_val.jsonl"]:
    path = os.path.join(HERE, "data", name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            used |= {json.loads(l)["messages"][1]["content"] for l in f}
rows = [r for r in rows if float(r["average_test_score"]) == 1.0 and judged_ok(r["llm_judgement"])
        and r["input"].strip() not in used]
random.seed(2026)
random.shuffle(rows)
rows = rows[:N_QUESTIONS]
print(f"{len(rows)} questions (verified pool, excluding the {len(used)} used for oci_*)")

done = {}
if os.path.exists(RAW):
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:  # a line cut off by stopping the run mid-write: that question is redone
                continue
            done[r["id"]] = r
todo = [r for r in rows if r["id"] not in done]
print(f"{len(done)} already sampled (resuming), {len(todo)} to go")

# ----------------------------------------------------------------------------
# sandboxed test runner
SANDBOX = ['unshare', '--mount', '--net', '--pid', '--fork', '--mount-proc', '--propagation', 'private',
           'sh', '-c',
           # inside the new namespaces: drop the Windows / host-shared mounts, then run as nobody
           'for m in $(awk \'$3=="9p" || $3=="drvfs" || $3=="virtiofs" {print $2}\' /proc/mounts | sort -r); '
           'do umount -l "$m" 2>/dev/null; done; cd "$1" && exec setpriv --reuid=65534 --regid=65534 '
           '--clear-groups --inh-caps=-all sh -c "ulimit -v 2000000; exec timeout 10 /usr/bin/python3 -I prog.py"',
           'sandbox']


def parse_tests(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return ast.literal_eval(raw)  # a Python-repr list in some rows


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len) if blocks else None


def with_aliases(code, tests):
    """Many questions never name the function, but their tests call it by the reference
    solution's name (`longest_palindrome` vs the model's `longest_palindromic_substring`) - 85% of
    the failures in a first sample were that NameError, not wrong answers. Alias each name the tests
    call but the answer doesn't define to the answer's closest top-level function (or class, for a
    capitalised name); the behaviour still has to pass the tests."""
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return code
    funcs = [n.name for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in body if isinstance(n, ast.ClassDef)]
    defined = set(funcs) | set(classes)
    defined |= {t.id for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    defined |= {a.asname or a.name.split(".")[0] for n in body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    called = set()
    for t in tests:
        try:
            called |= {n.func.id for n in ast.walk(ast.parse(t.strip()))
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        except SyntaxError:
            pass
    aliases = []
    for name in sorted(called - defined - set(dir(builtins))):
        pool = classes if name[:1].isupper() else funcs
        match = difflib.get_close_matches(name, pool, n=1, cutoff=0.0)
        if match:
            aliases.append(f"{name} = {match[0]}")
    return code + ("\n\n" + "\n".join(aliases) if aliases else "")


def run_tests(code, tests):
    d = tempfile.mkdtemp(prefix="sd_")
    try:
        os.chmod(d, 0o777)  # nobody may write scratch files next to the program
        with open(os.path.join(d, "prog.py"), "w") as f:
            f.write(with_aliases(code, tests) + "\n\n" + "\n".join(t.strip() for t in tests) + "\n")
        os.chmod(os.path.join(d, "prog.py"), 0o644)
        r = subprocess.run(SANDBOX + [d], stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------
# generate (or, with RESCORE=1, only re-run the tests on the saved samples - e.g. after a harness fix)
tokenizer = load_tokenizer()
pool = ThreadPoolExecutor(8)
if RESCORE:
    tests_by_id = {r["id"]: parse_tests(r["unit_tests"]) for r in rows}
    recs = [rec for rec in done.values() if rec["id"] in tests_by_id]
    jobs = []
    for rec in recs:
        for x in rec["samples"]:
            # a sample that ran into the token cap is truncated: never passes
            capped = len(tokenizer(x["text"], add_special_tokens=False)["input_ids"]) >= MAX_NEW_TOKENS - 1
            code = None if capped else extract_code(x["text"])
            jobs.append((x, pool.submit(run_tests, code, tests_by_id[rec["id"]]) if code else None))
    before = sum(x["passed"] for x, _ in jobs)
    for x, fut in jobs:
        x["passed"] = bool(fut and fut.result())
    print(f"rescored {len(jobs)} samples: {before} -> {sum(x['passed'] for x, _ in jobs)} passing")
    with open(RAW, "w", encoding="utf-8") as raw:
        for rec in done.values():
            raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
else:
    configure_runtime("cuda")
    tokenizer.padding_side = "left"
    print(f"loading {MODEL_ID}")
    model = load_with_adapter(None).eval()
    # any of these in the output means the answer ended on its own (padding only follows a stop)
    eos_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id, tokenizer.pad_token_id}


    def prompt_of(r):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"].strip()}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


    todo.sort(key=lambda r: len(r["input"]))  # similar lengths per batch -> less padding
    q_per_batch = max(1, GEN_BATCH // K)
    t0, n_tokens = time.time(), 0
    with open(RAW, "a", encoding="utf-8") as raw:
        for b in range(0, len(todo), q_per_batch):
            batch = todo[b:b + q_per_batch]
            enc = tokenizer([prompt_of(r) for r in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
            with torch.no_grad():
                # Qwen's recommended non-thinking sampling settings
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7,
                                     top_p=0.8, top_k=20, num_return_sequences=K,
                                     pad_token_id=tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:].tolist()
            jobs = []
            for qi, r in enumerate(batch):
                tests = parse_tests(r["unit_tests"])
                for s in range(K):
                    ids = gen[qi * K + s]
                    finished = any(t in eos_ids for t in ids)  # hit the token cap -> truncated, reject
                    ids = [t for t in ids if t != tokenizer.pad_token_id]
                    n_tokens += len(ids)
                    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    code = extract_code(text) if finished else None
                    jobs.append((qi, text, pool.submit(run_tests, code, tests) if code else None))
            samples = [[] for _ in batch]
            for qi, text, fut in jobs:
                samples[qi].append({"text": text, "passed": bool(fut and fut.result())})
            for r, s in zip(batch, samples):
                rec = {"id": r["id"], "input": r["input"].strip(), "samples": s}
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[r["id"]] = rec
            raw.flush()
            n_done = len(done)
            n_pass = sum(x["passed"] for rec in done.values() for x in rec["samples"])
            print(f"{n_done}/{len(rows)} questions | sample pass rate {n_pass / (n_done * K):.3f} | "
                  f"{n_tokens / (time.time() - t0):.0f} gen tok/s", flush=True)

# ----------------------------------------------------------------------------
# build the SFT set from the passing samples
hist = [0] * (K + 1)
examples = []
for rec in (done[r["id"]] for r in rows if r["id"] in done):
    passing = list(dict.fromkeys(x["text"] for x in rec["samples"] if x["passed"]))  # distinct, in order
    hist[sum(x["passed"] for x in rec["samples"])] += 1
    for text in passing[:KEEP_PER_QUESTION]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rec["input"]},
                    {"role": "assistant", "content": text}]
        full = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        if len(tokenizer(full, add_special_tokens=False)["input_ids"]) <= MAX_SEQ_LEN:
            examples.append((rec["id"], messages))
print("questions by number of passing samples (0.." + str(K) + "):", hist)

ids = sorted({i for i, _ in examples})
random.shuffle(ids)
val_ids = set(ids[:max(1, int(len(ids) * VAL_FRACTION))])  # split by question: no leakage between samples
for name, keep in [("sd_train.jsonl", lambda i: i not in val_ids), ("sd_val.jsonl", lambda i: i in val_ids)]:
    rows_out = [m for i, m in examples if keep(i)]
    with open(os.path.join(HERE, "data", name), "w", encoding="utf-8") as f:
        for messages in rows_out:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows_out)} -> data/{name}")

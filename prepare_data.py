"""
Build the SFT set from nvidia/OpenCodeInstruct (CC-BY-4.0): Python coding questions answered by
Qwen2.5-Coder-32B-Instruct, each with LLM-generated unit tests, their pass rate, and an LLM-judge
score. One parquet shard (100k rows) is far more than a laptop run needs, so this keeps only the
verified answers and samples from those:

  - average_test_score == 1.0     every unit test passes (~35% of rows)
  - judge scores all >= MIN_JUDGE requirement conformance, logical correctness, edge cases
  - fits MAX_SEQ_LEN tokens       dropped, not truncated - a cut-off answer teaches cut-off code

Writes data/oci_train.jsonl and data/oci_val.jsonl in train.py's chat format.

    python prepare_data.py <shard.parquet>     (run inside WSL: needs pyarrow + the tokenizer)
"""
import json
import os
import random
import sys

import pyarrow.parquet as pq

from model import MODEL_ID, load_tokenizer

SHARD = sys.argv[1]
N_TRAIN = int(os.environ.get("N_TRAIN", 3000))
N_VAL = int(os.environ.get("N_VAL", 150))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 1024))
MIN_JUDGE = int(os.environ.get("MIN_JUDGE", 4))
SYSTEM = "You are Cody, a concise and precise coding assistant."  # same default as chat.py
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

rows = pq.read_table(SHARD, columns=["id", "input", "output", "llm_judgement", "average_test_score"]).to_pylist()
print(f"{len(rows)} rows in {SHARD}")


def judged_ok(raw):
    try:
        return all(v["score"] >= MIN_JUDGE for v in json.loads(raw).values())
    except (ValueError, KeyError, TypeError):
        return False


rows = [r for r in rows if float(r["average_test_score"]) == 1.0]
print(f"{len(rows)} pass all unit tests")
rows = [r for r in rows if judged_ok(r["llm_judgement"])]
print(f"{len(rows)} also have judge scores >= {MIN_JUDGE}")

random.seed(1337)
random.shuffle(rows)
tokenizer = load_tokenizer()
print(f"tokenizer: {MODEL_ID}")


def to_messages(r):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["input"].strip()},
            {"role": "assistant", "content": r["output"].strip()}]


kept, too_long, lengths = [], 0, []
for r in rows:  # tokenize only as many as needed
    messages = to_messages(r)
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if n > MAX_SEQ_LEN:
        too_long += 1
        continue
    kept.append(messages)
    lengths.append(n)
    if len(kept) == N_TRAIN + N_VAL:
        break
print(f"kept {len(kept)} (skipped {too_long} longer than {MAX_SEQ_LEN} tokens on the way); "
      f"tokens/example mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}, total {sum(lengths)}")

val, train = kept[:N_VAL], kept[N_VAL:]
for name, data in [("oci_train.jsonl", train), ("oci_val.jsonl", val)]:
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        for messages in data:
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    print(f"wrote {len(data)} -> data/{name}")

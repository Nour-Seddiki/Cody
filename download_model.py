"""Download the base model to C:\\Users\\<you>\\models\\<name> with resumable, retry-forever HTTP.

Kept out of the repo folder on purpose: here it lives in OneDrive, which would upload the ~9 GB of weights.
Not snapshot_download: on this connection (drops for minutes at a time) its hf_xet backend stalled,
and its plain-HTTP path deleted a 2.7 GB partial file when DNS failed (2026-10-09). Here each file
goes to <file>.part, every retry resumes from its current size with a Range request, and a failure
never deletes anything. Re-running the script picks up where it stopped.

    python download_model.py [repo_id]      # default Qwen/Qwen3.5-4B
"""
import fnmatch
import hashlib
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_url

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
PATTERNS = ["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE"]
target = Path.home() / "models" / MODEL_ID.split("/")[-1]
target.mkdir(parents=True, exist_ok=True)


def with_retries(fn, what):
    delay = 5
    while True:
        try:
            return fn()
        except (httpx.HTTPError, OSError) as e:
            print(f"\n{what}: {type(e).__name__}: {e} - retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)


info = with_retries(lambda: HfApi().model_info(MODEL_ID, files_metadata=True), "listing files")
files = [s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in PATTERNS)]
total = sum(s.size or 0 for s in files)
print(f"{MODEL_ID} @ {info.sha[:8]}: {len(files)} files, {total / 1e9:.2f} GB -> {target}")


def fetch(sibling):
    dest = target / sibling.rfilename
    if dest.exists() and dest.stat().st_size == sibling.size:
        return
    part = dest.with_name(dest.name + ".part")
    url = hf_hub_url(MODEL_ID, sibling.rfilename, revision=info.sha)

    def attempt():
        have = part.stat().st_size if part.exists() else 0
        if have < sibling.size:
            headers = {"Range": f"bytes={have}-"} if have else {}
            timeout = httpx.Timeout(30.0, read=60.0)
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as r:
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the Range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    t0, done = time.time(), have
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        mbps = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"\r  {sibling.rfilename}: {done / 1e9:.2f}/{sibling.size / 1e9:.2f} GB "
                              f"({mbps:.1f} MB/s)   ", end="", flush=True)
        if part.stat().st_size != sibling.size:
            raise OSError(f"size {part.stat().st_size} != expected {sibling.size}")
        if sibling.lfs:  # weights: check the hash, since the file was stitched from resumed requests
            h = hashlib.sha256()
            with open(part, "rb") as f:
                while block := f.read(1 << 24):
                    h.update(block)
            if h.hexdigest() != sibling.lfs.sha256:
                part.unlink()
                raise OSError("sha256 mismatch - deleted, downloading again")
        part.replace(dest)

    with_retries(attempt, sibling.rfilename)
    print(f"\r  {sibling.rfilename}: done ({sibling.size / 1e9:.2f} GB)" + " " * 20, flush=True)


for s in sorted(files, key=lambda s: s.size or 0):  # small files first
    fetch(s)
print(f"downloaded {MODEL_ID} -> {target}")

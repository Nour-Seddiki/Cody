# Run a Cody script inside WSL Ubuntu: .\run.ps1 train.py [args...]   (works from any directory)
#
# Why WSL: on 2026-10-09 Smart App Control started blocking torch's unsigned torch_python.dll, so
# torch no longer imports on Windows here. Linux is not subject to it, and also gets the Triton
# kernels (flash-linear-attention) for Qwen3.5's Gated DeltaNet layers, which have no Windows build.
# Env: ~/venvs/cody in Ubuntu (see README).
param(
    [Parameter(Mandatory)][string]$Script,
    [Parameter(ValueFromRemainingArguments)][string[]]$ScriptArgs
)

# weights: a copy on the WSL disk loads in seconds; reading the C:\Users\<you>\models copy that
# download_model.py fetched goes through /mnt/c (9P) and takes ~2.5 min; else the Hugging Face id
$wslHome = wsl.exe -d Ubuntu --exec bash -c 'echo $HOME'
if (-not $env:MODEL_ID) {
    $wslModel = "$wslHome/models/Qwen3.5-4B"
    wsl.exe -d Ubuntu --exec test -f "$wslModel/model.safetensors-00001-of-00002.safetensors" -a -f "$wslModel/model.safetensors-00002-of-00002.safetensors"
    if ($LASTEXITCODE -eq 0) {
        $env:MODEL_ID = $wslModel
    } elseif (Test-Path "$env:USERPROFILE\models\Qwen3.5-4B\model.safetensors.index.json") {
        $env:MODEL_ID = "/mnt/c/Users/$env:USERNAME/models/Qwen3.5-4B"
    }
}

# WSL runs on the Windows GPU driver, where the CUDA cache spills into shared RAM instead of
# OOMing (see runtime.py) - cap it below VRAM unless told otherwise
if (-not $env:CUDA_MEM_FRACTION) { $env:CUDA_MEM_FRACTION = "0.85" }

# config env vars to pass through to Linux (set them in PowerShell before calling this script);
# paths must be relative to the repo root or Linux paths
$forward = "MODEL_ID TRAIN_FILE VAL_FILE VAL_FRACTION OUT_DIR MAX_SEQ_LEN BATCH_SIZE GRAD_ACCUM_STEPS " +
           "EPOCHS MAX_STEPS LR WARMUP_RATIO LORA_R LORA_ALPHA LORA_DROPOUT QUANTIZE EVAL_EVERY " +
           "LOG_EVERY ADAPTER PROMPT MAX_NEW_TOKENS SYSTEM CUDA_MEM_FRACTION N_TRAIN N_VAL MIN_JUDGE " +
           "N_PROBLEMS GEN_BATCH TASKS N_QUESTIONS K KEEP_PER_QUESTION RESCORE"
$set = $forward.Split(" ") | Where-Object { Test-Path "env:$_" }
$env:WSLENV = (@($set) + @($env:WSLENV) | Where-Object { $_ }) -join ":"

# keep Windows out of sleep / Modern Standby while the run lasts (standby freezes WSL too);
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED, released when this process exits
Add-Type -Namespace Cody -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[void][Cody.Power]::SetThreadExecutionState([uint32]2147483651)  # 0x80000003 (PS 5.1 parses the hex literal as a negative Int32)

# --exec: no shell between Windows and Linux, so nothing expands $ or mangles quotes on the way
$dir = wsl.exe -d Ubuntu --exec wslpath -a ($PSScriptRoot -replace '\\', '/')
wsl.exe -d Ubuntu --cd $dir --exec "$wslHome/venvs/cody/bin/python" -u $Script @ScriptArgs

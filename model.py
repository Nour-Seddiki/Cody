"""
Loading the Qwen base model (4-bit NF4 by default = QLoRA) and attaching / loading LoRA adapters.
Shared by train.py and chat.py.

Qwen3.5 checkpoints are vision-language models; AutoModelForCausalLM maps them to
Qwen3_5ForCausalLM, which loads only the text weights (the vision tower is skipped), so it costs no
VRAM here.
"""
import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")

# every projection in the full-attention, linear-attention (Gated DeltaNet) and MLP blocks; the
# DeltaNet's in_proj_a / in_proj_b are left out - they project to num_heads (32) outputs only
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(quantize=True):
    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)


def add_lora(model, r=16, alpha=32, dropout=0.05):
    # not peft's prepare_model_for_kbit_training: it upcasts every non-quantized weight to fp32,
    # and with Qwen3.5's 248k x 2560 (tied) embedding that is +1.2 GiB of VRAM for nothing - the
    # base stays frozen. Checkpointing + input grads is the part of it LoRA actually needs.
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                        bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def load_with_adapter(adapter_dir, quantize=True):
    model = load_base(quantize)
    return PeftModel.from_pretrained(model, adapter_dir) if adapter_dir else model

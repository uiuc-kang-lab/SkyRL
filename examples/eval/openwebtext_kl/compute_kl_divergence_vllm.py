"""Compute KL divergence between two LoRA adapters using vLLM for generation.

This variant uses vLLM for fast generation (model_a) and HuggingFace for logit
extraction from both models.  LoRA adapters are merged into the base model
before loading.

Usage
-----
uv run --extra fsdp scripts/compute_kl_divergence_vllm.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_a /path/to/lora_a \
    --lora_b /path/to/lora_b \
    --dataset data/prompts.parquet \
    --max_gen_length 1024 \
    --tp_size 1 \
    --batch_size 16
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams

from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits, masked_mean
from skyrl.utils.tok import get_tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(source: str, prompt_key: str = "prompt") -> hf_datasets.Dataset:
    ext = os.path.splitext(source)[-1].lower()
    if ext == ".parquet":
        ds = hf_datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
    elif ext in (".json", ".jsonl"):
        ds = hf_datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
    else:
        dataset_name, has_split, split = source.partition(":")
        ds_dict = hf_datasets.load_dataset(path=dataset_name, keep_in_memory=True)
        split = split if has_split else "train"
        ds = ds_dict[split]

    if prompt_key not in ds.column_names:
        raise ValueError(
            f"Dataset does not contain prompt key '{prompt_key}'. "
            f"Available columns: {ds.column_names}"
        )
    return ds


# ---------------------------------------------------------------------------
# KL computation (same as single-GPU variant)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_kl_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    kl_type: str = "exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KL(model_a || model_b) per token.

    Returns (kl, mask) each of shape (batch, seq_len - 1).
    """
    if kl_type == "exact":
        log_probs_a = F.log_softmax(logits_a[:, :-1].float(), dim=-1)
        log_probs_b = F.log_softmax(logits_b[:, :-1].float(), dim=-1)
        probs_a = log_probs_a.exp()
        kl = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)
        mask = response_mask[:, 1:]
        kl = kl * mask
        return kl, mask
    else:
        labels = input_ids[:, 1:]
        lp_a = logprobs_from_logits(logits_a[:, :-1].float(), labels)
        lp_b = logprobs_from_logits(logits_b[:, :-1].float(), labels)
        mask = response_mask[:, 1:]
        kl = compute_approx_kl(lp_a, lp_b, loss_mask=mask, kl_estimator_type=kl_type)
        return kl, mask


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _validate_lora_adapter(lora_path: str) -> None:
    """Validate that a LoRA adapter directory looks correct."""
    if not (Path(lora_path) / "adapter_config.json").exists():
        raise ValueError(
            f"No adapter_config.json found in {lora_path}. "
            "Pass the LoRA adapter directory directly."
        )


def _patch_qwen35_config_for_vllm(merged_path: str) -> None:
    """Restructure Qwen3.5 text config as VL config for vLLM compatibility.

    vLLM normalizes Qwen3_5ForCausalLM -> Qwen3_5ForConditionalGeneration (the
    VL architecture) and then expects Qwen3_5Config (VL format) rather than
    Qwen3_5TextConfig.  After PEFT merge, save_pretrained writes the text-only
    config.  This function wraps it in the VL structure so vLLM's multimodal
    registry type-check passes.
    """
    config_path = os.path.join(merged_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("model_type") != "qwen3_5_text":
        return

    top_level_keys = {
        "architectures", "model_type", "transformers_version",
        "torch_dtype", "auto_map", "_commit_hash",
        "_attn_implementation_autoset",
    }
    text_config = {k: v for k, v in cfg.items() if k not in top_level_keys}
    text_config["model_type"] = "qwen3_5_text"

    vl_config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": text_config,
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 27,
            "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 3584,
            "num_position_embeddings": 2304,
        },
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
    }
    if "transformers_version" in cfg:
        vl_config["transformers_version"] = cfg["transformers_version"]

    with open(config_path, "w") as f:
        json.dump(vl_config, f, indent=2)

    logger.info("Patched config.json to VL format for vLLM compatibility")


def merge_lora_to_disk(base_model_path: str, lora_path: str, output_dir: str) -> str:
    """Merge a LoRA adapter into the base model and save to disk.

    Returns the path to the merged model directory.
    """
    _validate_lora_adapter(lora_path)
    logger.info(f"Merging LoRA adapter {lora_path} into {base_model_path}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, lora_path)
    merged = merged.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = get_tokenizer(base_model_path, trust_remote_code=True, padding_side="left")
    tokenizer.save_pretrained(output_dir)

    _patch_qwen35_config_for_vllm(output_dir)

    logger.info(f"Merged model saved to {output_dir}")

    del base, merged
    torch.cuda.empty_cache()
    return output_dir


def load_hf_model(
    base_model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lora_path: str | None = None,
) -> AutoModelForCausalLM:
    """Load a model with optional LoRA merge, for logit extraction."""
    logger.info(f"Loading HF model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if lora_path is not None:
        _validate_lora_adapter(lora_path)
        logger.info(f"Applying LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged and unloaded")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    tmp_dir = tempfile.mkdtemp(prefix="kl_vllm_")
    try:
        return _run_inner(args, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_inner(args: argparse.Namespace, tmp_dir: str) -> dict:
    tokenizer = get_tokenizer(args.base_model, trust_remote_code=True, padding_side="left")

    model_a_desc = f"{args.base_model} + LoRA({args.lora_a})" if args.lora_a else args.base_model
    model_b_desc = f"{args.base_model} + LoRA({args.lora_b})" if args.lora_b else args.base_model

    # ---- Resolve model_a path for vLLM (needs on-disk weights) ----
    if args.lora_a is not None:
        vllm_model_path = merge_lora_to_disk(args.base_model, args.lora_a, os.path.join(tmp_dir, "model_a"))
    else:
        vllm_model_path = args.base_model

    # ---- vLLM for generation ----
    logger.info(f"Initializing vLLM engine for model_a: {model_a_desc}")
    llm = LLM(
        model=vllm_model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_prompt_length + args.max_gen_length,
        seed=args.seed,
        # Required for Qwen3.5 models whose config is patched to VL format;
        # tells vLLM to skip loading the vision tower.
        language_model_only=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0.0,
        top_p=1.0,
        max_tokens=args.max_gen_length,
    )

    # ---- Load dataset ----
    ds = load_dataset(args.dataset, prompt_key=args.prompt_key)
    if args.num_samples is not None and args.num_samples < len(ds):
        import random
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(ds)), args.num_samples))
        ds = ds.select(indices)
    logger.info(f"Dataset: {len(ds)} prompts")

    # Tokenize
    all_prompt_ids: list[list[int]] = []
    for i in range(len(ds)):
        prompt = ds[i][args.prompt_key]
        if isinstance(prompt, list):
            ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) <= args.max_prompt_length:
            all_prompt_ids.append(ids)
    logger.info(f"After length filter: {len(all_prompt_ids)} prompts")

    # ---- Generate with vLLM ----
    logger.info("Generating responses with model_a via vLLM...")
    vllm_outputs = llm.generate(
        prompt_token_ids=all_prompt_ids,
        sampling_params=sampling_params,
    )

    # Extract prompt + response token sequences
    sequences: list[tuple[list[int], list[int]]] = []
    for out in vllm_outputs:
        prompt_ids = list(out.prompt_token_ids)
        response_ids = list(out.outputs[0].token_ids)
        if len(response_ids) > 0:
            sequences.append((prompt_ids, response_ids))
    logger.info(f"Generated {len(sequences)} non-empty responses")

    # Free vLLM memory
    del llm
    torch.cuda.empty_cache()

    # ---- Load HF models for logit extraction ----
    device = torch.device(f"cuda:0")
    dtype = torch.bfloat16

    logger.info("Loading model_a (HF) for logit extraction...")
    model_a_hf = load_hf_model(args.base_model, device, dtype, lora_path=args.lora_a)

    logger.info("Loading model_b (HF) for logit extraction...")
    model_b_hf = load_hf_model(args.base_model, device, dtype, lora_path=args.lora_b)

    # ---- Compute KL in batches ----
    batch_size = args.batch_size
    all_kl_means = []
    all_kl_maxs = []
    total_tokens = 0
    total_kl_sum = 0.0

    for start in tqdm(range(0, len(sequences), batch_size), desc="Computing KL"):
        batch = sequences[start : start + batch_size]
        pad_id = tokenizer.pad_token_id

        # Build padded tensors: left-pad to uniform length
        full_seqs = [p + r for p, r in batch]
        prompt_lens = [len(p) for p, _ in batch]
        max_len = max(len(s) for s in full_seqs)

        padded_ids = []
        attn_masks = []
        resp_masks = []

        for seq, plen in zip(full_seqs, prompt_lens):
            pad_len = max_len - len(seq)
            padded_ids.append([pad_id] * pad_len + seq)
            attn_masks.append([0] * pad_len + [1] * len(seq))
            resp_masks.append([0.0] * (pad_len + plen) + [1.0] * (len(seq) - plen))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)
        response_mask = torch.tensor(resp_masks, dtype=torch.float, device=device)

        with torch.no_grad():
            logits_a = model_a_hf(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_b = model_b_hf(input_ids=input_ids, attention_mask=attention_mask).logits

        kl, mask = compute_kl_from_logits(logits_a, logits_b, input_ids, response_mask, args.kl_type)

        seq_token_counts = mask.sum(dim=-1)
        seq_kl_sum = kl.sum(dim=-1)

        for i in range(len(batch)):
            n_tok = seq_token_counts[i].item()
            if n_tok > 0:
                all_kl_means.append(seq_kl_sum[i].item() / n_tok)
                all_kl_maxs.append((kl[i] * mask[i]).max().item())
                total_tokens += n_tok
                total_kl_sum += seq_kl_sum[i].item()

        del logits_a, logits_b, kl, mask
        torch.cuda.empty_cache()

    if not all_kl_means:
        logger.error("No valid sequences processed")
        return {}

    kl_means_arr = np.array(all_kl_means)
    kl_maxs_arr = np.array(all_kl_maxs)

    metrics = {
        "base_model": args.base_model,
        "lora_a": args.lora_a,
        "lora_b": args.lora_b,
        "model_a": model_a_desc,
        "model_b": model_b_desc,
        "kl_type": args.kl_type,
        "num_sequences": len(all_kl_means),
        "total_response_tokens": int(total_tokens),
        "dataset_mean_kl_per_token": float(total_kl_sum / total_tokens),
        "sequence_mean_kl": {
            "mean": float(kl_means_arr.mean()),
            "std": float(kl_means_arr.std()),
            "median": float(np.median(kl_means_arr)),
            "min": float(kl_means_arr.min()),
            "max": float(kl_means_arr.max()),
            "p5": float(np.percentile(kl_means_arr, 5)),
            "p25": float(np.percentile(kl_means_arr, 25)),
            "p75": float(np.percentile(kl_means_arr, 75)),
            "p95": float(np.percentile(kl_means_arr, 95)),
        },
        "sequence_max_kl": {
            "mean": float(kl_maxs_arr.mean()),
            "std": float(kl_maxs_arr.std()),
            "max": float(kl_maxs_arr.max()),
        },
    }

    logger.info("=" * 60)
    logger.info(f"KL Divergence: {args.kl_type.upper()}")
    logger.info(f"  KL(model_a || model_b)")
    logger.info(f"  model_a: {model_a_desc}")
    logger.info(f"  model_b: {model_b_desc}")
    logger.info(f"  sequences: {metrics['num_sequences']}")
    logger.info(f"  total response tokens: {metrics['total_response_tokens']}")
    logger.info("-" * 60)
    logger.info(f"  Dataset mean KL/token:  {metrics['dataset_mean_kl_per_token']:.6f}")
    logger.info(f"  Sequence mean KL/token: {metrics['sequence_mean_kl']['mean']:.6f} +/- {metrics['sequence_mean_kl']['std']:.6f}")
    logger.info(f"  Median:                 {metrics['sequence_mean_kl']['median']:.6f}")
    logger.info(f"  [p5, p95]:              [{metrics['sequence_mean_kl']['p5']:.6f}, {metrics['sequence_mean_kl']['p95']:.6f}]")
    logger.info(f"  Max token KL (avg):     {metrics['sequence_max_kl']['mean']:.6f}")
    logger.info("=" * 60)

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute KL divergence between two models (vLLM generation + HF logits)"
    )
    p.add_argument("--base_model", required=True, help="Path or HF name of the base model")
    p.add_argument("--lora_a", default=None, help="Path to LoRA adapter for model A (omit to use base model)")
    p.add_argument("--lora_b", default=None, help="Path to LoRA adapter for model B (omit to use base model)")
    p.add_argument("--dataset", required=True, help=".parquet, .json/.jsonl, or HF dataset[:split]")
    p.add_argument("--prompt_key", default="prompt")
    p.add_argument("--max_gen_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tp_size", type=int, default=1, help="vLLM tensor parallel size")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    p.add_argument("--kl_type", choices=["exact", "k1", "k2", "k3", "abs"], default="exact")
    p.add_argument("--output_path", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

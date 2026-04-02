"""Compute KL divergence of output token distributions between two LoRA adapters.

Given a base model, two LoRA adapter paths, and a dataset of prompts, this script:
1. Loads the base model and applies each LoRA adapter to create model_a and model_b.
2. Tokenizes each prompt and generates responses from model_a.
3. Computes the full next-token distribution (logits) from both models over
   the generated response tokens.
4. Reports per-token, per-sequence, and dataset-level KL(model_a || model_b).

Supports both *exact* KL over the full vocabulary and the *approximate* KL
estimators from SkyRL (k1, k2, k3, abs).

Usage
-----
# Two LoRA adapters on the same base model:
uv run --extra fsdp scripts/compute_kl_divergence.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_a /path/to/lora_adapter_a \
    --lora_b /path/to/lora_adapter_b \
    --dataset data/prompts.parquet \
    --max_gen_length 512 \
    --batch_size 4

# Base model vs a LoRA adapter (omit --lora_a to use base as model_a):
uv run --extra fsdp scripts/compute_kl_divergence.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_b /path/to/lora_adapter \
    --dataset data/prompts.parquet
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import datasets as hf_datasets
import torch
import torch.nn.functional as F
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits, masked_mean
from skyrl.utils.tok import get_tokenizer


# ---------------------------------------------------------------------------
# Dataset loading (mirrors SkyRL's PromptDataset conventions)
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
# Model helpers
# ---------------------------------------------------------------------------

def _validate_lora_adapter(lora_path: str) -> None:
    """Validate that a LoRA adapter directory looks correct."""
    if not (Path(lora_path) / "adapter_config.json").exists():
        raise ValueError(
            f"No adapter_config.json found in {lora_path}. "
            "Pass the LoRA adapter directory directly."
        )


def load_model(
    base_model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
) -> AutoModelForCausalLM:
    """Load a model, optionally applying and merging a LoRA adapter.

    Args:
        base_model_path: HF model name or local path for the base model.
        device: Target device.
        dtype: Model dtype.
        lora_path: If provided, load and merge this LoRA adapter into the base.

    Returns:
        The (possibly merged) model in eval mode on ``device``.
    """
    logger.info(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
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


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompt_ids_batch: list[list[int]],
    max_gen_length: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate responses and return (input_ids, attention_mask, response_mask).

    input_ids includes both prompt and generated tokens, left-padded.
    response_mask marks which positions are generated tokens (1) vs prompt/pad (0).
    """
    # Left-pad prompts
    max_prompt_len = max(len(ids) for ids in prompt_ids_batch)
    pad_id = tokenizer.pad_token_id
    padded = []
    prompt_lengths = []
    for ids in prompt_ids_batch:
        pad_len = max_prompt_len - len(ids)
        padded.append([pad_id] * pad_len + ids)
        prompt_lengths.append(max_prompt_len)  # after padding, prompt ends at max_prompt_len

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = (input_ids != pad_id).long()

    # Generate
    if temperature <= 0:
        gen_kwargs = {"do_sample": False}
    else:
        gen_kwargs = {"do_sample": True, "temperature": temperature, "top_p": 1.0}

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_gen_length,
        pad_token_id=pad_id,
        **gen_kwargs,
    )

    # outputs shape: (batch, prompt_len + gen_len)
    full_ids = outputs
    full_attn = (full_ids != pad_id).long()

    # Build response mask: 1 for generated tokens, 0 for prompt + padding
    response_mask = torch.zeros_like(full_ids, dtype=torch.float)
    response_mask[:, max_prompt_len:] = full_attn[:, max_prompt_len:]

    return full_ids, full_attn, response_mask


@torch.no_grad()
def get_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass to get logits. Returns shape (batch, seq_len, vocab_size)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits


# ---------------------------------------------------------------------------
# KL computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_exact_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute exact token-level KL(P_a || P_b) over the full vocabulary.

    KL(P||Q) = sum_x P(x) * (log P(x) - log Q(x))

    Args:
        logits_a: (batch, seq_len, vocab) from model A.
        logits_b: (batch, seq_len, vocab) from model B.
        response_mask: (batch, seq_len) binary mask for response tokens.
            KL is computed using the *preceding* position's distribution to
            predict the *current* token, so we shift: use logits[:, t-1, :]
            for position t.  The mask marks response positions.

    Returns:
        Per-token KL of shape (batch, seq_len), masked by response_mask.
    """
    # Shift: logits at position t predict token t+1.
    # For response tokens starting at position s, we use logits at s-1..end-1.
    # We compute KL at every position and mask later.
    log_probs_a = F.log_softmax(logits_a[:, :-1].float(), dim=-1)
    log_probs_b = F.log_softmax(logits_b[:, :-1].float(), dim=-1)
    probs_a = log_probs_a.exp()

    # KL(P_a || P_b) = sum_x P_a(x) * (log P_a(x) - log P_b(x))
    kl = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)  # (batch, seq_len-1)

    # Align mask: response_mask[:, 1:] since logits[:, t] predicts token t+1
    mask = response_mask[:, 1:]
    kl = kl * mask
    return kl, mask


@torch.no_grad()
def compute_approx_kl_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    kl_estimator_type: str = "k3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute approximate KL using only the log-prob of the selected token.

    This mirrors SkyRL's compute_approx_kl from ppo_utils.py.

    Args:
        logits_a: (batch, seq_len, vocab) from model A.
        logits_b: (batch, seq_len, vocab) from model B.
        input_ids: (batch, seq_len) token IDs (prompt + response).
        response_mask: (batch, seq_len) binary mask.
        kl_estimator_type: One of "k1", "k2", "k3", "abs".

    Returns:
        (kl, mask) each of shape (batch, seq_len-1).
    """
    # Labels for logprobs: token at position t+1 is predicted by logits at t
    labels = input_ids[:, 1:]  # (batch, seq_len-1)
    shifted_logits_a = logits_a[:, :-1]
    shifted_logits_b = logits_b[:, :-1]

    log_probs_a = logprobs_from_logits(shifted_logits_a.float(), labels)
    log_probs_b = logprobs_from_logits(shifted_logits_b.float(), labels)

    mask = response_mask[:, 1:]
    kl = compute_approx_kl(log_probs_a, log_probs_b, loss_mask=mask, kl_estimator_type=kl_estimator_type)
    return kl, mask


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Resolve model descriptions for logging
    model_a_desc = f"{args.base_model} + LoRA({args.lora_a})" if args.lora_a else args.base_model
    model_b_desc = f"{args.base_model} + LoRA({args.lora_b})" if args.lora_b else args.base_model

    # Load tokenizer from the base model
    tokenizer = get_tokenizer(args.base_model, trust_remote_code=True, padding_side="left")

    # Load models (base + optional LoRA)
    model_a = load_model(args.base_model, device, dtype, lora_path=args.lora_a)
    model_b = load_model(args.base_model, device, dtype, lora_path=args.lora_b)

    # Load dataset
    ds = load_dataset(args.dataset, prompt_key=args.prompt_key)
    if args.num_samples is not None and args.num_samples < len(ds):
        import random
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(ds)), args.num_samples))
        ds = ds.select(indices)
    logger.info(f"Dataset: {len(ds)} prompts")

    # Tokenize prompts
    all_prompt_ids: list[list[int]] = []
    for i in range(len(ds)):
        prompt = ds[i][args.prompt_key]
        if isinstance(prompt, list):
            # Chat format (list of dicts)
            ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
        else:
            # Plain text
            ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) <= args.max_prompt_length:
            all_prompt_ids.append(ids)

    logger.info(f"After filtering by max_prompt_length={args.max_prompt_length}: {len(all_prompt_ids)} prompts")

    # Process in batches
    batch_size = args.batch_size
    all_kl_means = []
    all_kl_maxs = []
    all_seq_kl = []
    total_tokens = 0
    total_kl_sum = 0.0

    for start in tqdm(range(0, len(all_prompt_ids), batch_size), desc="Computing KL"):
        batch_ids = all_prompt_ids[start : start + batch_size]

        # Generate from model_a
        full_ids, full_attn, response_mask = generate_responses(
            model_a, tokenizer, batch_ids, args.max_gen_length, args.temperature, device
        )

        # Check that we have generated tokens
        n_response_tokens = response_mask.sum().item()
        if n_response_tokens == 0:
            logger.warning(f"Batch {start}: no response tokens generated, skipping")
            continue

        # Get logits from both models
        logits_a = get_logits(model_a, full_ids, full_attn)
        logits_b = get_logits(model_b, full_ids, full_attn)

        # Compute KL
        if args.kl_type == "exact":
            kl, mask = compute_exact_kl(logits_a, logits_b, response_mask)
        else:
            kl, mask = compute_approx_kl_from_logits(
                logits_a, logits_b, full_ids, response_mask, kl_estimator_type=args.kl_type
            )

        # Per-sequence KL (mean over response tokens)
        seq_token_counts = mask.sum(dim=-1)  # (batch,)
        seq_kl_sum = kl.sum(dim=-1)  # (batch,)

        for i in range(len(batch_ids)):
            n_tok = seq_token_counts[i].item()
            if n_tok > 0:
                mean_kl = seq_kl_sum[i].item() / n_tok
                max_kl = (kl[i] * mask[i]).max().item() if mask[i].sum() > 0 else 0.0
                all_kl_means.append(mean_kl)
                all_kl_maxs.append(max_kl)
                all_seq_kl.append(seq_kl_sum[i].item())
                total_tokens += n_tok
                total_kl_sum += seq_kl_sum[i].item()

        # Free memory
        del logits_a, logits_b, kl, mask, full_ids, full_attn, response_mask
        torch.cuda.empty_cache()

    # Aggregate metrics
    if not all_kl_means:
        logger.error("No valid sequences processed")
        return {}

    import numpy as np
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

    # Print summary
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

    # Save results
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute KL divergence of output token distributions between two models"
    )
    p.add_argument("--base_model", required=True, help="Path or HF name of the base model")
    p.add_argument("--lora_a", default=None, help="Path to LoRA adapter for model A (omit to use base model)")
    p.add_argument("--lora_b", default=None, help="Path to LoRA adapter for model B (omit to use base model)")
    p.add_argument("--dataset", required=True, help="Dataset path: .parquet, .json/.jsonl, or HF dataset name[:split]")
    p.add_argument("--prompt_key", default="prompt", help="Key in dataset for the prompt field")
    p.add_argument("--max_gen_length", type=int, default=512, help="Maximum generation length")
    p.add_argument("--max_prompt_length", type=int, default=4096, help="Maximum prompt length (filter longer)")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for generation and forward passes")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    p.add_argument("--num_samples", type=int, default=None, help="Subsample N prompts from the dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0, help="GPU device index")
    p.add_argument(
        "--kl_type",
        choices=["exact", "k1", "k2", "k3", "abs"],
        default="exact",
        help="KL computation method: 'exact' for full-vocab KL, or approx estimators from SkyRL",
    )
    p.add_argument("--output_path", default=None, help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

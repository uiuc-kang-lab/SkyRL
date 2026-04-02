"""Compute KL divergence between a LoRA-finetuned model and its base model.

Given a base model, a LoRA adapter, and a dataset of prompts, this script:
1. Loads the base model twice: once with LoRA merged (model_a) and once
   without (model_b).
2. Generates responses from the LoRA model (model_a).
3. Computes the full next-token distribution (logits) from both models over
   the generated response tokens.
4. Reports per-token, per-sequence, and dataset-level KL(lora || base).

Supports both *exact* KL over the full vocabulary and the *approximate* KL
estimators from SkyRL (k1, k2, k3, abs).

Usage
-----
uv run --extra fsdp -m examples.eval.openwebtext_kl.compute_kl_divergence \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora /path/to/lora_adapter \
    --dataset data/prompts.parquet \
    --max_gen_length 512 \
    --batch_size 4
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import datasets as hf_datasets
import torch
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl
from skyrl.backends.skyrl_train.utils.torch_utils import logprobs_from_logits
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





# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    lora_desc = f"{args.base_model} + LoRA({args.lora})"
    base_desc = args.base_model

    # Load tokenizer from the base model
    tokenizer = get_tokenizer(args.base_model, trust_remote_code=True, padding_side="left")

    # Load dataset
    ds = load_dataset(args.dataset, prompt_key=args.prompt_key)
    if args.num_samples is not None and args.num_samples < len(ds):
        import random
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(ds)), args.num_samples))
        ds = ds.select(indices)
    logger.info(f"Dataset: {len(ds)} prompts")

    # Tokenize prompts (truncate to max_prompt_length)
    all_prompt_ids: list[list[int]] = []
    n_truncated = 0
    for i in range(len(ds)):
        prompt = ds[i][args.prompt_key]
        if isinstance(prompt, list):
            # Chat format (list of dicts)
            ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
        else:
            # Plain text
            ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) > args.max_prompt_length:
            ids = ids[: args.max_prompt_length]
            n_truncated += 1
        all_prompt_ids.append(ids)

    logger.info(f"Tokenized {len(all_prompt_ids)} prompts ({n_truncated} truncated to {args.max_prompt_length} tokens)")

    batch_size = args.batch_size
    pad_id = tokenizer.pad_token_id

    # ---- Phase 1: Generate with LoRA model and collect its logprobs ----
    logger.info("Loading LoRA model for generation + logprob extraction...")
    model_a = load_model(args.base_model, device, dtype, lora_path=args.lora)

    # Generate responses and store (prompt_ids, response_ids) pairs
    sequences: list[tuple[list[int], list[int]]] = []
    for start in tqdm(range(0, len(all_prompt_ids), batch_size), desc="Generating"):
        batch_ids = all_prompt_ids[start : start + batch_size]
        full_ids, full_attn, response_mask = generate_responses(
            model_a, tokenizer, batch_ids, args.max_gen_length, args.temperature, device
        )
        # Extract per-sequence token lists
        max_prompt_len = max(len(ids) for ids in batch_ids)
        for i, ids in enumerate(batch_ids):
            pad_len = max_prompt_len - len(ids)
            prompt_part = ids  # original unpadded prompt
            # response = non-pad tokens after the prompt
            resp_start = max_prompt_len
            resp_ids = []
            for j in range(resp_start, full_ids.shape[1]):
                tok = full_ids[i, j].item()
                if tok == pad_id:
                    break
                resp_ids.append(tok)
            if resp_ids:
                sequences.append((prompt_part, resp_ids))
        del full_ids, full_attn, response_mask
        torch.cuda.empty_cache()

    logger.info(f"Generated {len(sequences)} non-empty responses")

    def _build_batch_tensors(batch):
        full_seqs = [p + r for p, r in batch]
        prompt_lens = [len(p) for p, _ in batch]
        max_len = max(len(s) for s in full_seqs)
        padded_ids, attn_masks, resp_masks = [], [], []
        for seq, plen in zip(full_seqs, prompt_lens):
            pl = max_len - len(seq)
            padded_ids.append([pad_id] * pl + seq)
            attn_masks.append([0] * pl + [1] * len(seq))
            resp_masks.append([0.0] * (pl + plen) + [1.0] * (len(seq) - plen))
        return (
            torch.tensor(padded_ids, dtype=torch.long, device=device),
            torch.tensor(attn_masks, dtype=torch.long, device=device),
            torch.tensor(resp_masks, dtype=torch.float, device=device),
        )

    def _collect_logprobs(model, label):
        all_lp, all_masks = [], []
        for start in tqdm(range(0, len(sequences), batch_size), desc=f"Logprobs ({label})"):
            batch = sequences[start : start + batch_size]
            input_ids, attention_mask, response_mask = _build_batch_tensors(batch)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            labels = input_ids[:, 1:]
            lp = logprobs_from_logits(logits[:, :-1].float(), labels)
            mask = response_mask[:, 1:]
            all_lp.append(lp.cpu())
            all_masks.append(mask.cpu())
            del logits, lp
            torch.cuda.empty_cache()
        return all_lp, all_masks

    # Collect LoRA model logprobs
    lp_a_batches, mask_batches = _collect_logprobs(model_a, "lora")
    del model_a
    torch.cuda.empty_cache()

    # ---- Phase 2: Collect base model logprobs ----
    logger.info("Loading base model for logprob extraction...")
    model_b = load_model(args.base_model, device, dtype, lora_path=None)
    lp_b_batches, _ = _collect_logprobs(model_b, "base")
    del model_b
    torch.cuda.empty_cache()

    # ---- Phase 3: Compute KL from stored logprobs ----
    all_kl_means = []
    all_kl_maxs = []
    total_tokens = 0
    total_kl_sum = 0.0

    for lp_a, lp_b, masks in zip(lp_a_batches, lp_b_batches, mask_batches):
        kl = compute_approx_kl(lp_a, lp_b, loss_mask=masks, kl_estimator_type=args.kl_type)
        seq_token_counts = masks.sum(dim=-1)
        seq_kl_sum = kl.sum(dim=-1)

        for i in range(kl.shape[0]):
            n_tok = seq_token_counts[i].item()
            if n_tok > 0:
                all_kl_means.append(seq_kl_sum[i].item() / n_tok)
                all_kl_maxs.append((kl[i] * masks[i]).max().item())
                total_tokens += n_tok
                total_kl_sum += seq_kl_sum[i].item()

    # Aggregate metrics
    if not all_kl_means:
        logger.error("No valid sequences processed")
        return {}

    import numpy as np
    kl_means_arr = np.array(all_kl_means)
    kl_maxs_arr = np.array(all_kl_maxs)

    metrics = {
        "base_model": args.base_model,
        "lora": args.lora,
        "model_a": lora_desc,
        "model_b": base_desc,
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
    logger.info(f"  KL(lora || base)")
    logger.info(f"  lora:  {lora_desc}")
    logger.info(f"  base:  {base_desc}")
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
    p.add_argument("--lora", required=True, help="Path to LoRA adapter directory")
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
        choices=["k1", "k2", "k3", "abs"],
        default="k3",
        help="Approximate KL estimator from SkyRL (k3 = Schulman's approximation)",
    )
    p.add_argument("--output_path", default=None, help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

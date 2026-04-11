"""
Quantize a custom LoRA adapter (v parameters) at multiple levels.

All v parameters are flattened into a single vector, uniformly quantized
to the given levels, then reconstructed into per-tensor safetensors.

The compression metrics (entropy, prefix length, generalization bound)
follow the methodology in model_layer_selection/quantize_adapter.py.

Usage:
    uv run python3 -m examples.train.eurus_rl_math.quantize_custom_lora \
        --adapter ../models/Qwen3.5-4B_tinylora_opd/custom_lora_params.safetensors \
        --levels 2 3 4 5 8 11 \
        --out-dir ../models/Qwen3.5-4B_tinylora_opd \
        --train-error 0.40 --n 450000
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def uniform_quantize(param_vector: np.ndarray, levels: int) -> np.ndarray:
    """Replace each value with the centroid of its uniform bin."""
    largest = max(float(np.max(param_vector)), float(abs(np.min(param_vector))))
    bin_edges = np.linspace(-largest - 1e-6, largest + 1e-6, levels + 1)
    centroids = (bin_edges[:-1] + bin_edges[1:]) / 2
    symbols = np.digitize(param_vector, bin_edges) - 1
    symbols = np.clip(symbols, 0, levels - 1)
    return centroids[symbols]


def compression_result(param_vector: np.ndarray, levels: int):
    """Compute entropy-coded message length for the quantized parameter vector."""
    import scipy.stats

    largest = max(float(np.max(param_vector)), float(abs(np.min(param_vector))))
    bin_edges = np.linspace(-largest - 1e-6, largest + 1e-6, levels + 1)
    symbols = np.digitize(param_vector, bin_edges) - 1
    symbols = np.clip(symbols, 0, levels - 1)
    probs = np.array([np.mean(symbols == i) for i in range(levels)])
    H = float(scipy.stats.entropy(probs, base=2))
    D = len(param_vector)
    coded = math.ceil(D * H) + 1
    codebook = levels * 16        # 16 bits per centroid
    prob_bits = math.ceil(math.log2(D)) * levels
    msg = coded + codebook + prob_bits
    prefix = msg + 2 * math.log2(msg)
    return H, prefix


def subsampling_bound(prefix_len: float, train_error: float, n: int, epsilon: float = 0.05) -> float:
    """PAC-Bayes / subsampling generalization bound."""
    div = prefix_len * math.log(2)
    return train_error + math.sqrt((div - math.log(epsilon / 2)) / (2 * n))


def main():
    parser = argparse.ArgumentParser(description="Quantize custom LoRA v parameters")
    parser.add_argument("--adapter", required=True,
                        help="Path to custom_lora_params.safetensors")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 8, 11],
                        help="Quantization levels to produce")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same directory as adapter)")
    parser.add_argument("--train-error", type=float, default=0.40,
                        help="Training error rate for generalization bound")
    parser.add_argument("--n", type=int, default=450_000,
                        help="Training set size for generalization bound")
    args = parser.parse_args()

    src = Path(args.adapter)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent

    # Load source tensors
    tensors = load_file(str(src))
    keys = list(tensors.keys())

    # Flatten all v params into one vector
    shapes = {k: tensors[k].shape for k in keys}
    dtypes = {k: tensors[k].dtype for k in keys}
    parts = [tensors[k].float().numpy().ravel() for k in keys]
    sizes = [p.size for p in parts]
    param_vector = np.concatenate(parts)

    print(f"Source adapter : {src}")
    print(f"Parameters     : {len(keys)} tensors, D = {len(param_vector):,}")
    print(f"Output dir     : {out_dir}")
    print()

    # Preview: show a slice of the first tensor
    preview_key = keys[0]
    preview_orig = parts[0][:8]
    print(f"Preview tensor : {preview_key}")
    print(f"  original[:8] : {np.array2string(preview_orig, precision=6, suppress_small=True)}")
    for lvl in sorted(args.levels):
        q_preview = uniform_quantize(preview_orig, lvl)
        print(f"  levels={lvl:<3}[:8] : {np.array2string(q_preview, precision=6, suppress_small=True)}")
    print()

    print(f"{'levels':>8}  {'entropy':>10}  {'prefix_len':>14}  {'bound':>10}  output")
    print("-" * 80)

    for lvl in sorted(args.levels):
        q_vec = uniform_quantize(param_vector, lvl)
        H, prefix = compression_result(param_vector, lvl)
        b = subsampling_bound(prefix, args.train_error, args.n)

        # Reconstruct per-tensor dict
        q_tensors = {}
        offset = 0
        for k, sz in zip(keys, sizes):
            chunk = q_vec[offset: offset + sz].reshape(shapes[k])
            q_tensors[k] = torch.from_numpy(chunk).to(dtypes[k])
            offset += sz

        # Save quantized adapter
        out_name = f"custom_lora_params_q{lvl}.safetensors"
        out_path = out_dir / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(q_tensors, str(out_path))

        print(f"{lvl:>8}  {H:>10.4f}  {prefix:>14,.1f}  {b:>10.6f}  {out_name}")


if __name__ == "__main__":
    main()

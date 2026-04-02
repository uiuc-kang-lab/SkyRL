"""Pluggable approximation schemes for custom LoRA.

Each scheme defines how to:
1. Initialize fixed buffers from the original weight matrix.
2. Compute the LoRA output efficiently during forward pass.
3. Reconstruct the full delta_W matrix (for merge/unmerge during weight sync).

New schemes are registered in SCHEME_REGISTRY and selected by name in config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor


class ApproximationScheme(ABC):
    """Base class for custom LoRA approximation schemes."""

    @abstractmethod
    def initialize(
        self,
        weight: Tensor,
        svd_rank: int,
        num_coefficients: int,
        seed: int,
    ) -> Dict[str, Tensor]:
        """Compute fixed buffers from original weight.

        Args:
            weight: Original weight matrix (out_features, in_features).
            svd_rank: Truncation rank r for SVD.
            num_coefficients: Number of trainable scalars u.
            seed: Deterministic seed for random components.

        Returns:
            Dict mapping buffer names to tensors.  All buffers are registered
            as persistent and will be saved in checkpoints and synced by FSDP.
        """

    @abstractmethod
    def forward(self, x: Tensor, v: Tensor, buffers: Dict[str, Tensor]) -> Tensor:
        """Compute LoRA output efficiently (no full delta_W materialisation).

        Args:
            x: Input tensor (..., in_features).
            v: Trainable coefficients (num_coefficients,).
            buffers: Fixed buffers from ``initialize``.

        Returns:
            LoRA output (..., out_features).
        """

    @abstractmethod
    def compute_delta(self, v: Tensor, buffers: Dict[str, Tensor]) -> Tensor:
        """Reconstruct full delta_W (out_features, in_features).

        Used only for merge/unmerge during weight sync to inference engines.
        """

    def merge_into(
        self, weight: Tensor, v: Tensor, buffers: Dict[str, Tensor]
    ) -> None:
        """Add delta_W into *weight* in-place.

        Default implementation materializes the full delta.  Schemes can
        override this with a chunked version that avoids the full (d, k)
        allocation.
        """
        delta = self.compute_delta(v, buffers)
        weight.data.add_(delta.to(weight.dtype))
        del delta


class SVDRandomProjection(ApproximationScheme):
    """Default scheme: SVD basis + random projection coefficients.

    delta_W = U_scaled @ M @ V^T
    where M = sum_i v_i * P_i   (r x r matrix)

    U_scaled = U @ diag(S)  from truncated SVD of W.
    P is a fixed random (u, r, r) tensor.
    v is the only trainable parameter (u scalars).
    """

    def initialize(
        self,
        weight: Tensor,
        svd_rank: int,
        num_coefficients: int,
        seed: int,
    ) -> Dict[str, Tensor]:
        out_features, in_features = weight.shape
        dtype = weight.dtype
        device = weight.device

        # Truncated SVD — use float32 for numerical stability.
        # Eagerly delete the float32 copy and intermediate results
        # to minimise peak GPU memory during sequential layer init.
        #
        # torch.svd_lowrank uses a randomised algorithm internally
        # (random projection → QR → SVD).  We run it inside fork_rng
        # which snapshots and restores both CPU and CUDA RNG states,
        # so the seeded SVD call cannot disturb downstream randomness.
        # Fork the weight's CUDA device (if any) so that GPU random
        # state is also isolated.
        w_f32 = weight.float()
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(seed)
            U, S, V = torch.svd_lowrank(w_f32, q=svd_rank)
        del w_f32
        # U: (out_features, r), S: (r,), V: (in_features, r)

        # Fold singular values into U, then drop full-precision intermediates
        U_scaled = (U * S.unsqueeze(0)).to(dtype)  # (out_features, r)
        V = V.to(dtype)  # (in_features, r)
        del U, S  # free float32 originals

        # Fixed random projection basis — deterministic from seed.
        # Generated on CPU to avoid fragmenting GPU memory, then moved.
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        P = torch.randn(num_coefficients, svd_rank, svd_rank, generator=rng, dtype=dtype)
        # Normalise each P_i so that ||P_i||_F = 1 for stable initialisation
        P = P / (P.norm(dim=(-2, -1), keepdim=True) + 1e-8)

        return {
            "U_scaled": U_scaled.to(device),
            "V": V.to(device),
            "P": P.to(device),
        }

    def forward(self, x: Tensor, v: Tensor, buffers: Dict[str, Tensor]) -> Tensor:
        # FSDP mixed precision may cast buffers to buffer_dtype (e.g. float32)
        # while x is in param_dtype (e.g. bf16).  Cast buffers to x's dtype
        # to avoid wasteful implicit upcasting in matmuls.
        compute_dtype = x.dtype
        U_scaled = buffers["U_scaled"].to(compute_dtype)  # (out_features, r)
        V = buffers["V"].to(compute_dtype)  # (in_features, r)
        P = buffers["P"].to(compute_dtype)  # (u, r, r)
        v = v.to(compute_dtype)

        # M = sum_i v_i * P_i → (r, r)
        M = torch.einsum("u, uij -> ij", v, P)

        # Efficient chain of small matmuls (never materialise full delta_W):
        #   x: (..., in_features)
        #   h1 = x @ V        → (..., r)
        #   h2 = h1 @ M^T     → (..., r)
        #   out = h2 @ U_scaled^T → (..., out_features)
        h = x @ V  # (..., r)
        h = h @ M.T  # (..., r)
        out = h @ U_scaled.T  # (..., out_features)
        return out

    def compute_delta(self, v: Tensor, buffers: Dict[str, Tensor]) -> Tensor:
        U_scaled = buffers["U_scaled"]  # (out_features, r)
        V = buffers["V"]  # (in_features, r)
        P = buffers["P"]  # (u, r, r)

        M = torch.einsum("u, uij -> ij", v, P)  # (r, r)
        # delta_W = U_scaled @ M @ V^T → (out_features, in_features)
        delta_W = U_scaled @ M @ V.T
        return delta_W

    def merge_into(
        self, weight: Tensor, v: Tensor, buffers: Dict[str, Tensor]
    ) -> None:
        """Add delta_W into *weight* without materializing the full matrix.

        Processes the addition in row-chunks to keep peak memory at
        ``chunk_rows * in_features`` instead of ``out_features * in_features``.
        Falls back to the dense path for small weights where chunking overhead
        would dominate.
        """
        U_scaled = buffers["U_scaled"]  # (out_features, r)
        V = buffers["V"]  # (in_features, r)
        P = buffers["P"]  # (u, r, r)

        M = torch.einsum("u, uij -> ij", v, P)  # (r, r)
        MV_T = M @ V.T  # (r, in_features)

        out_features = weight.shape[0]
        # Only chunk when the full delta would exceed ~32 MB (in bf16)
        full_size_bytes = out_features * weight.shape[1] * weight.element_size()
        if full_size_bytes <= 32 * 1024 * 1024:
            weight.data.add_((U_scaled @ MV_T).to(weight.dtype))
            return

        # Process in row-chunks to avoid materializing the full (d, k) delta
        chunk_rows = max(1, (32 * 1024 * 1024) // (weight.shape[1] * weight.element_size()))
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            weight.data[start:end].add_(
                (U_scaled[start:end] @ MV_T).to(weight.dtype)
            )


# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

SCHEME_REGISTRY: Dict[str, type] = {
    "svd_random_projection": SVDRandomProjection,
}


def get_scheme(name: str) -> ApproximationScheme:
    """Instantiate an approximation scheme by name."""
    if name not in SCHEME_REGISTRY:
        raise ValueError(
            f"Unknown custom LoRA scheme '{name}'. "
            f"Available: {list(SCHEME_REGISTRY.keys())}"
        )
    return SCHEME_REGISTRY[name]()

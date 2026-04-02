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
            Dict mapping buffer names to tensors.  Buffers whose names start
            with ``_ephemeral_`` will be registered as non-persistent (not saved
            in checkpoints, regenerated from seed).
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

        # Truncated SVD — use float32 for numerical stability
        w_f32 = weight.float()
        U, S, V = torch.svd_lowrank(w_f32, q=svd_rank)
        # U: (out_features, r), S: (r,), V: (in_features, r)

        # Fold singular values into U
        U_scaled = (U * S.unsqueeze(0)).to(dtype)  # (out_features, r)
        V = V.to(dtype)  # (in_features, r)

        # Fixed random projection basis — deterministic from seed
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        P = torch.randn(num_coefficients, svd_rank, svd_rank, generator=rng, dtype=dtype)
        # Normalise each P_i so that ||P_i||_F = 1 for stable initialisation
        P = P / (P.norm(dim=(-2, -1), keepdim=True) + 1e-8)

        return {
            "U_scaled": U_scaled.to(device),
            "V": V.to(device),
            # Prefix with _ephemeral_ → registered as non-persistent buffer
            "_ephemeral_P": P.to(device),
        }

    def forward(self, x: Tensor, v: Tensor, buffers: Dict[str, Tensor]) -> Tensor:
        U_scaled = buffers["U_scaled"]  # (out_features, r)
        V = buffers["V"]  # (in_features, r)
        P = buffers["_ephemeral_P"]  # (u, r, r)

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
        P = buffers["_ephemeral_P"]  # (u, r, r)

        M = torch.einsum("u, uij -> ij", v, P)  # (r, r)
        # delta_W = U_scaled @ M @ V^T → (out_features, in_features)
        delta_W = U_scaled @ M @ V.T
        return delta_W


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

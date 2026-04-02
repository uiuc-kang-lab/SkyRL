"""CustomLoraLinear — drop-in replacement for nn.Linear with extreme compression.

Only ``self.v`` (a vector of u scalars) is trainable.  All other tensors
(original weight, SVD factors, random projection) are frozen buffers.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schemes import ApproximationScheme, get_scheme

logger = logging.getLogger(__name__)





class CustomLoraLinear(nn.Module):
    """Linear layer with custom LoRA parameterisation.

    The forward pass computes::

        out = F.linear(x, weight, bias) + scheme.forward(x, v, buffers)

    where ``v`` is the *only* trainable parameter.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        svd_rank: int,
        num_coefficients: int,
        seed: int,
        scheme_name: str = "svd_random_projection",
    ) -> None:
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.svd_rank = svd_rank
        self.num_coefficients = num_coefficients
        self.seed = seed
        self.scheme_name = scheme_name

        # Frozen base weight & bias — keep as regular parameters with grad off
        self.weight = original_linear.weight
        self.weight.requires_grad_(False)
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad_(False)
        else:
            self.bias = None

        # Instantiate scheme
        self.scheme: ApproximationScheme = get_scheme(scheme_name)

        # Compute fixed buffers (SVD, random projection, etc.)
        is_meta = self.weight.is_meta
        if not is_meta:
            buffers = self.scheme.initialize(
                self.weight.data, svd_rank, num_coefficients, seed
            )
        else:
            # Meta-tensor init path: create placeholder meta buffers.
            # Real values will be populated via FSDP sync_module_states.
            buffers = self._meta_placeholders(svd_rank, num_coefficients)

        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=True)

        # The ONLY trainable parameter
        self.v = nn.Parameter(torch.zeros(num_coefficients, dtype=self.weight.dtype))

        self._merged = False

        # Cached buffer dict — avoids re-creating a dict on every forward call.
        # Invalidated by _invalidate_buffer_cache() if buffers ever change.
        self._buffer_cache: dict | None = None

    def _meta_placeholders(self, svd_rank: int, num_coefficients: int) -> dict:
        """Create meta-device placeholder buffers matching expected shapes."""
        dtype = self.weight.dtype
        return {
            "U_scaled": torch.empty(self.out_features, svd_rank, dtype=dtype, device="meta"),
            "V": torch.empty(self.in_features, svd_rank, dtype=dtype, device="meta"),
            "P": torch.empty(
                num_coefficients, svd_rank, svd_rank, dtype=dtype, device="meta"
            ),
        }

    def _get_buffers(self) -> dict:
        """Return registered buffers as a dict, using a cache to avoid
        repeated dict construction on the training hot path."""
        if self._buffer_cache is None:
            self._buffer_cache = {
                name: buf for name, buf in self.named_buffers(recurse=False)
            }
        return self._buffer_cache

    def _invalidate_buffer_cache(self) -> None:
        self._buffer_cache = None

    # Override register_buffer to invalidate cache
    def register_buffer(self, name, tensor, persistent=True):
        super().register_buffer(name, tensor, persistent=persistent)
        self._buffer_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.scheme.forward(x, self.v, self._get_buffers())
        return base_out + lora_out

    def add_delta_inplace(self, full_weight: torch.Tensor) -> None:
        """Add delta_W to *full_weight* **in-place**.

        FSDP-safe: operates on the already-gathered full tensor produced by
        ``FSDPWeightExtractor`` (which is a fresh allocation from
        ``full_tensor().to(dtype).detach().contiguous()``).  The module's
        own ``weight`` parameter (a flat FSDP shard outside forward) is
        never read or written.

        Zero extra GPU allocation beyond the ``(r, in_features)``
        intermediate inside ``merge_into``.
        """
        self.scheme.merge_into(full_weight, self.v, self._get_buffers())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"svd_rank={self.svd_rank}, num_coefficients={self.num_coefficients}, "
            f"scheme={self.scheme_name}"
        )

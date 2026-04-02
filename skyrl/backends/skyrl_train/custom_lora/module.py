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

EPHEMERAL_PREFIX = "_ephemeral_"


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
            persistent = not name.startswith(EPHEMERAL_PREFIX)
            self.register_buffer(name, tensor, persistent=persistent)

        # The ONLY trainable parameter
        self.v = nn.Parameter(torch.zeros(num_coefficients, dtype=self.weight.dtype))

        self._merged = False

    def _meta_placeholders(self, svd_rank: int, num_coefficients: int) -> dict:
        """Create meta-device placeholder buffers matching expected shapes."""
        dtype = self.weight.dtype
        return {
            "U_scaled": torch.empty(self.out_features, svd_rank, dtype=dtype, device="meta"),
            "V": torch.empty(self.in_features, svd_rank, dtype=dtype, device="meta"),
            "_ephemeral_P": torch.empty(
                num_coefficients, svd_rank, svd_rank, dtype=dtype, device="meta"
            ),
        }

    def _get_buffers(self) -> dict:
        """Collect registered buffers into a dict for the scheme."""
        return {name: buf for name, buf in self.named_buffers(recurse=False)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = self.scheme.forward(x, self.v, self._get_buffers())
        return base_out + lora_out

    def merge(self) -> None:
        """Merge delta_W into the base weight (for inference weight sync).

        Saves a copy of the original weight so that ``unmerge`` can restore
        it exactly, avoiding the numerical drift of add-then-subtract in
        low-precision dtypes (e.g. bf16).
        """
        if self._merged:
            return
        self._weight_backup = self.weight.data.clone()
        delta = self.scheme.compute_delta(self.v, self._get_buffers())
        self.weight.data.add_(delta.to(self.weight.dtype))
        self._merged = True

    def unmerge(self) -> None:
        """Restore the original base weight saved during ``merge``."""
        if not self._merged:
            return
        self.weight.data.copy_(self._weight_backup)
        del self._weight_backup
        self._merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"svd_rank={self.svd_rank}, num_coefficients={self.num_coefficients}, "
            f"scheme={self.scheme_name}"
        )

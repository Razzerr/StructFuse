"""Triangle multiplicative update (AF2 Evoformer Alg. 11 i 12).

Captures geometric consistency: jeżeli kontakty i-k i j-k istnieją, to
kontakt i-j jest bardziej prawdopodobny. Single biggest contribution
udokumentowana w AF2 / OpenFold / RoseTTAFold ablations.

Wersje:
- outgoing: c[i, j] = sum_k (a[i, k] * b[j, k])
- incoming: c[i, j] = sum_k (a[k, i] * b[k, j])

Safety:
- zero-init output_proj -> krok 0 jest identity residual (baseline parity)
- pair_mask zero'ujemy PRZED sumowaniem -> padding nie zatruwa valid pairs
- c_triangle << d_pair (default 32 vs 128) dla memory (OpenFold pattern)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, d_pair: int, c_triangle: int = 32, outgoing: bool = True):
        super().__init__()
        self.d_pair = d_pair
        self.c_triangle = c_triangle
        self.outgoing = outgoing

        self.layer_norm_in = nn.LayerNorm(d_pair)
        self.left_proj = nn.Linear(d_pair, c_triangle)
        self.right_proj = nn.Linear(d_pair, c_triangle)
        self.left_gate = nn.Linear(d_pair, c_triangle)
        self.right_gate = nn.Linear(d_pair, c_triangle)
        self.output_gate = nn.Linear(d_pair, d_pair)
        self.layer_norm_out = nn.LayerNorm(c_triangle)
        self.output_proj = nn.Linear(c_triangle, d_pair)

        # Zero-init output projection => identity residual start.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # Gate bias = -1 => sigmoid(-1) ≈ 0.27, gentle start.
        nn.init.constant_(self.output_gate.bias, -1.0)

    def forward(self, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            z: (B, L, L, C) pair features (channel-last).
            pair_mask: (B, L, L) or (B, 1, L, L) valid-pair mask.
        Returns:
            (B, L, L, C) updated pair features.
        """
        if pair_mask.dim() == 4:
            pair_mask = pair_mask.squeeze(1)
        mask = pair_mask.to(z.dtype).unsqueeze(-1)  # (B, L, L, 1)

        z_norm = self.layer_norm_in(z)
        a = torch.sigmoid(self.left_gate(z_norm)) * self.left_proj(z_norm)
        b = torch.sigmoid(self.right_gate(z_norm)) * self.right_proj(z_norm)

        a = a * mask
        b = b * mask

        if self.outgoing:
            c = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            c = torch.einsum("bkic,bkjc->bijc", a, b)

        c = self.layer_norm_out(c)
        g = torch.sigmoid(self.output_gate(z_norm))
        out = g * self.output_proj(c)
        return z + out

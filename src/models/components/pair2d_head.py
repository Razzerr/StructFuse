import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models.components.fusion_strategies import get_fusion_strategy


@functools.lru_cache(maxsize=8)
def relpos_buckets(L: int, device: torch.device, cuts: tuple = (0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64)) -> torch.Tensor:
    """
    Returns a relative-position encoding: (R, L, L), where R = len(cuts)+1.
    Bin k means |i-j| ∈ (cuts[k-1], cuts[k]] with k=0 for |i-j| <= cuts[0].

    Results are cached (LRU, maxsize=32) to avoid recomputation while
    bounding GPU memory from stale entries.
    """
    idx = torch.arange(L, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()  # (L, L)
    edges = torch.tensor(cuts, device=device)
    bins = torch.bucketize(dist, edges)  # (L, L) in [0..len(cuts)]
    R = len(cuts) + 1
    oh = F.one_hot(bins.clamp_max(R - 1), num_classes=R)  # (L, L, R)
    rel = oh.permute(2, 0, 1).float()  # (R, L, L)
    return rel


class BasicBlock(nn.Module):
    def __init__(self, c, dropout=0.1, use_depthwise=False, dilation=1):
        super().__init__()
        # Option to use regular convolutions instead of depthwise separable
        # Regular convs can better mix row/column artifacts
        # Dilation allows larger receptive field without more parameters
        
        padding = dilation  # Keep same output size: padding = dilation for 3x3 kernel
        
        ## FROM: MobileNet/EfficientNet papers
        if use_depthwise:
            # Depthwise separable convolutions (more efficient)
            self.net = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=padding, dilation=dilation, groups=c),  # depthwise
                nn.Conv2d(c, c, 1),  # pointwise
                nn.GroupNorm(8, c),
                nn.ReLU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(c, c, 3, padding=padding, dilation=dilation, groups=c),
                nn.Conv2d(c, c, 1),
                nn.GroupNorm(8, c),
            )
        else:
            # Regular convolutions (better feature mixing)
            self.net = nn.Sequential(
                nn.Conv2d(c, c, 3, padding=padding, dilation=dilation),
                nn.GroupNorm(8, c),
                nn.ReLU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(c, c, 3, padding=padding, dilation=dilation),
                nn.GroupNorm(8, c),
            )

        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.net(x) + x)


class AxialAttentionBlock(nn.Module):
    """
    Axial attention block for 2D pairwise features.

    Each block applies row-attention, column-attention, an outer-product
    pairwise update, and FFN — all with **pre-norm** residual connections
    (standard in AlphaFold2 / OpenFold).

    Row and column attention use **separate** QKV projections so they
    can learn axis-specific transformations.

    The **outer-product update** breaks the additive f(i)+g(j)
    decomposition inherent to axial attention.  It derives per-residue
    features by averaging the pair matrix along each axis, computes
    their outer product, and projects back:

        a = proj_a( mean_j z[i,j] )          # (B, L, c)
        b = proj_b( mean_i z[i,j] )          # (B, L, c)
        Δz[i,j] = Linear( a[i] ⊗ b[j] )     # (B, L, L, C)

    This gives rank-c bilinear interactions so that every channel's
    update depends jointly on residues i *and* j, not on either one
    alone.  The subsequent FFN can then refine these pairwise features
    through cross-channel nonlinear mixing.

    Args:
        channels: Number of channels
        num_heads: Number of query attention heads (must divide channels)
        num_kv_heads: Number of KV heads for GQA (default: None = same as num_heads)
        axis: Which axis to attend: 0=both (legacy), 1=row only, 2=col only
        dropout: Dropout rate
        ffn_expansion: FFN expansion factor (default: 2)
        c_outer: Outer-product projection dimension (default: 8)
    """
    def __init__(
        self, 
        channels: int, 
        num_heads: int = 4,
        num_kv_heads: int = None,
        axis: int = 0,
        dropout: float = 0.1,
        ffn_expansion: int = 2,
        c_outer: int = 8,
    ):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.c_outer = c_outer
        self.axis = axis  # 0=both, 1=row, 2=col
        
        # GQA: fewer KV heads than query heads
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.num_kv_heads = num_kv_heads
        self.kv_group_size = num_heads // num_kv_heads
        kv_dim = num_kv_heads * self.head_dim
        
        # Only allocate projections for active axes
        if axis in (0, 1):
            self.q_row = nn.Linear(channels, channels, bias=False)
            self.kv_row = nn.Linear(channels, kv_dim * 2, bias=False)
            self.proj_row = nn.Linear(channels, channels)
            self.norm_row = nn.LayerNorm(channels)
        if axis in (0, 2):
            self.q_col = nn.Linear(channels, channels, bias=False)
            self.kv_col = nn.Linear(channels, kv_dim * 2, bias=False)
            self.proj_col = nn.Linear(channels, channels)
            self.norm_col = nn.LayerNorm(channels)

        # Pre-norm layers for outer product and FFN
        self.norm_outer = nn.LayerNorm(channels)
        self.norm_ffn = nn.LayerNorm(channels)

        # Outer-product update: breaks f(i)+g(j) additivity
        self.outer_a = nn.Linear(channels, c_outer)
        self.outer_b = nn.Linear(channels, c_outer)
        self.outer_out = nn.Sequential(
            nn.Linear(c_outer * c_outer, channels),
            nn.Dropout(dropout),
        )
        
        # Lightweight FFN (2x expansion)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ffn_expansion, channels),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _axial_attention(self, x, axis, pair_mask=None):
        """
        GQA attention along one axis.
        
        Args:
            x: (B, L, L, C) tensor (already normalized by caller)
            axis: 1 for row attention, 2 for column attention
            pair_mask: (B, 1, L, L) binary mask (1 = valid, 0 = padding), or None
        """
        B, L, _, C = x.shape
        q_proj = self.q_row if axis == 1 else self.q_col
        kv_proj = self.kv_row if axis == 1 else self.kv_col
        out_proj = self.proj_row if axis == 1 else self.proj_col
        
        # Rearrange based on axis
        if axis == 1:  # Row: attend across columns for each row
            x_seq = x.reshape(B * L, L, C)
        else:  # Column: attend across rows for each column
            x_seq = x.transpose(1, 2).reshape(B * L, L, C)
        
        N = B * L  # number of independent sequences
        
        # Q projection: full num_heads
        q = q_proj(x_seq).reshape(N, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (N, H, L, D)
        
        # KV projection: fewer heads (GQA)
        kv = kv_proj(x_seq).reshape(N, L, 2, self.num_kv_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, N, Hkv, L, D)
        k, v = kv[0], kv[1]
        
        # Expand KV heads to match Q heads if GQA
        if self.kv_group_size > 1:
            # (N, Hkv, L, D) → (N, Hkv, G, L, D) → (N, H, L, D)
            k = k[:, :, None].expand(N, self.num_kv_heads, self.kv_group_size, L, self.head_dim)
            k = k.reshape(N, self.num_heads, L, self.head_dim)
            v = v[:, :, None].expand(N, self.num_kv_heads, self.kv_group_size, L, self.head_dim)
            v = v.reshape(N, self.num_heads, L, self.head_dim)
        
        # Flash / fused SDPA — no attn_mask to allow FlashAttention-2 kernel
        drop_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop_p, is_causal=False
        )
        out = out.transpose(1, 2).reshape(N, L, C)
        out = out_proj(out)
        
        # Reshape back
        if axis == 1:
            out = out.reshape(B, L, L, C)
        else:
            out = out.reshape(B, L, L, C).transpose(1, 2)
        
        return out
        
    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, L, L) 2D pairwise features
            pair_mask: (B, 1, L, L) binary mask (1 = valid, 0 = padding)
        Returns:
            (B, C, L, L) attended features
        """
        B, C, L, _ = x.shape
        x_attn = x.permute(0, 2, 3, 1)  # (B, L, L, C)
        
        # Alternating or dual-axis attention based on self.axis
        if self.axis in (0, 1):
            x_attn = x_attn + self._axial_attention(self.norm_row(x_attn), axis=1, pair_mask=pair_mask)
        if self.axis in (0, 2):
            x_attn = x_attn + self._axial_attention(self.norm_col(x_attn), axis=2, pair_mask=pair_mask)

        # Outer-product update: z[i,j] += Linear(a[i] ⊗ b[j])
        # Breaks axial attention's additive f(i)+g(j) decomposition
        z = self.norm_outer(x_attn)

        if pair_mask is not None:
            # pair_mask: (B, 1, L, L) → row/col masks for masked mean
            # Row mask: valid columns for each row → (B, L, L, 1)
            row_m = pair_mask.squeeze(1).unsqueeze(-1)          # (B, L, L, 1)
            col_m = pair_mask.squeeze(1).transpose(1, 2).unsqueeze(-1)  # (B, L, L, 1)
            a = self.outer_a((z * row_m).sum(dim=2) / row_m.sum(dim=2).clamp(min=1))  # (B, L, c_outer)
            b = self.outer_b((z * col_m).sum(dim=1) / col_m.sum(dim=1).clamp(min=1))  # (B, L, c_outer)
        else:
            a = self.outer_a(z.mean(dim=2))   # (B, L, c_outer)
            b = self.outer_b(z.mean(dim=1))   # (B, L, c_outer)

        # Explicit outer product — avoids 5D einsum that Inductor can't lower
        outer = a[:, :, None, :, None] * b[:, None, :, None, :]  # (B,L,L,c,c)
        outer = outer.reshape(B, L, L, self.c_outer * self.c_outer)
        x_attn = x_attn + self.outer_out(outer)

        # FFN refines the now-pairwise (non-additive) features
        x_attn = x_attn + self.ffn(self.norm_ffn(x_attn))
        
        return x_attn.permute(0, 3, 1, 2)


class Pair2DHead(nn.Module):
    """
    2D head for contact prediction with configurable fusion and architecture.
    
    Args:
        d_pair: Dimension of pairwise features from PairFeatures
        width: Hidden dimension for processing blocks
        depth: Number of processing blocks
        rel_ch: Dimension of relative position embeddings
        fusion_strategy: "standard" (gated + aux) or "trufor" (cross-attention)
        fusion_num_heads: Number of attention heads for TruFor fusion
        fusion_reduction: Channel reduction factor for TruFor fusion
        head_type: "cnn", "dilated", or "axial" - architecture type for processing
        head_num_heads: Number of attention heads if head_type="axial"
        use_depthwise: Whether to use depthwise separable convs if head_type="cnn"
    """
    def __init__(
        self, 
        d_pair: int,
        width: int = 128, 
        depth: int = 8, 
        rel_ch: int = 14,
        fusion_strategy: str = "standard",
        fusion_num_heads: int = 8,
        fusion_reduction: int = 1,
        n_dist_bins: int = 0,
        n_ss_feat: int = 0,
        n_out: int = 1,  # 1 for binary contact, N for distogram
        head_type: str = "cnn",
        head_num_heads: int = 4,  # Reduced default for efficiency
        head_num_kv_heads: int = None,  # GQA: KV heads (None = same as head_num_heads)
        alternating_axial: bool = False,  # Alternating row/col instead of both per block
        use_depthwise: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.head_type = head_type
        self.use_checkpoint = use_checkpoint
        
        # Fusion strategy selection
        self.fusion_strategy = fusion_strategy
        self.fusion = get_fusion_strategy(
            strategy=fusion_strategy,
            d_pair=d_pair,
            d_rel=rel_ch,
            num_heads=fusion_num_heads,
            reduction=fusion_reduction,
            n_dist_bins=n_dist_bins,
            n_ss_feat=n_ss_feat,
        )
        
        # Input channels depend on fusion strategy output
        in_ch = self.fusion.out_channels
        
        # Input projection
        self.inp = nn.Conv2d(in_ch, width, 1)
        
        # Processing blocks - CNN, Dilated CNN, or Axial Attention
        if head_type == "axial":
            self.blocks = nn.ModuleList([
                AxialAttentionBlock(
                    width,
                    num_heads=head_num_heads,
                    num_kv_heads=head_num_kv_heads,
                    # alternating: even=row, odd=col; otherwise both axes
                    axis=(1 + (i % 2)) if alternating_axial else 0,
                )
                for i in range(depth)
            ])
        elif head_type == "dilated":
            # Dilated convolutions with exponentially increasing dilation
            # Provides larger receptive field without attention overhead
            dilations = [2 ** (i % 3) for i in range(depth)]  # [1, 2, 4, 1, 2, 4, ...]
            self.blocks = nn.Sequential(*[
                BasicBlock(width, use_depthwise=use_depthwise, dilation=dilations[i]) 
                for i in range(depth)
            ])
        elif head_type == "cnn":
            self.blocks = nn.Sequential(*[
                BasicBlock(width, use_depthwise=use_depthwise) 
                for _ in range(depth)
            ])
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'cnn', 'dilated', or 'axial'")
        
        # Output projection
        self.n_out = n_out
        self.out = nn.Conv2d(width, n_out, 1)
        if n_out == 1:
            # Prior probability init for binary contact: contacts ~5% of valid pairs.
            # bias = log(π/(1-π)) ≈ -2.94 so initial sigmoid ≈ 0.05.
            import math
            nn.init.constant_(self.out.bias, -math.log((1 - 0.05) / 0.05))

    def forward(self, pair_feat, prior, count, rel, esm_contacts, pair_mask=None, dist_bins=None, ss_feat=None, return_intermediates=False):
        # Apply fusion strategy
        fusion_out = self.fusion(
            pair_feat, prior, count, rel, esm_contacts,
            dist_bins=dist_bins, ss_feat=ss_feat,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            x, diag = fusion_out
        else:
            x = fusion_out
            diag = {}
        
        # Processing
        x = self.inp(x)
        if self.head_type == "axial":
            for i, block in enumerate(self.blocks):
                x_in = x
                if self.use_checkpoint and self.training:
                    x = grad_checkpoint(block, x, pair_mask, use_reentrant=False)
                else:
                    x = block(x, pair_mask=pair_mask)
                if return_intermediates:
                    in_norm = x_in.norm().item()
                    residual_norm = (x - x_in).norm().item()
                    diag[f"block{i}_in_norm"] = in_norm
                    diag[f"block{i}_residual_ratio"] = residual_norm / (in_norm + 1e-8)
        else:
            for i, block in enumerate(self.blocks):
                x_in = x
                x = block(x)
                if return_intermediates:
                    in_norm = x_in.norm().item()
                    residual_norm = (x - x_in).norm().item()
                    diag[f"block{i}_in_norm"] = in_norm
                    diag[f"block{i}_residual_ratio"] = residual_norm / (in_norm + 1e-8)

        logits = self.out(x)
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        if return_intermediates:
            diag["logits_mean"] = logits.mean().item()
            diag["logits_std"] = logits.std().item()
            diag["logits_min"] = logits.min().item()
            diag["logits_max"] = logits.max().item()
            return logits, diag

        return logits

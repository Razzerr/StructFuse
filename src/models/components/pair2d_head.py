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
        num_heads: Number of attention heads (must divide channels)
        dropout: Dropout rate
        ffn_expansion: FFN expansion factor (default: 2)
        c_outer: Outer-product projection dimension (default: 8)
    """
    def __init__(
        self, 
        channels: int, 
        num_heads: int = 4,
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
        
        # Separate projections for row and column attention
        self.qkv_row = nn.Linear(channels, channels * 3, bias=False)
        self.proj_row = nn.Linear(channels, channels)
        self.qkv_col = nn.Linear(channels, channels * 3, bias=False)
        self.proj_col = nn.Linear(channels, channels)

        # Pre-norm layers (applied BEFORE attention / FFN / outer product)
        self.norm_row = nn.LayerNorm(channels)
        self.norm_col = nn.LayerNorm(channels)
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
        Attention along one axis with axis-specific QKV.
        
        Args:
            x: (B, L, L, C) tensor (already normalized by caller)
            axis: 1 for row attention, 2 for column attention
            pair_mask: (B, 1, L, L) binary mask (1 = valid, 0 = padding), or None
        
        Note: We intentionally omit attn_mask from SDPA so the FlashAttention
        kernel can be used (FA2 does not support arbitrary bool masks).
        Padding positions are zeroed out in the residual path by the caller's
        pair_mask multiplication, and pre-norm ensures padding features are
        near-zero so softmax naturally de-weights them.
        """
        B, L, _, C = x.shape
        qkv_proj = self.qkv_row if axis == 1 else self.qkv_col
        out_proj = self.proj_row if axis == 1 else self.proj_col
        
        # Rearrange based on axis
        if axis == 1:  # Row: attend across columns for each row
            x_seq = x.reshape(B * L, L, C)
        else:  # Column: attend across rows for each column
            x_seq = x.transpose(1, 2).reshape(B * L, L, C)
        
        # QKV projection
        qkv = qkv_proj(x_seq).reshape(B * L, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*L, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash / fused SDPA — no attn_mask to allow FlashAttention-2 kernel
        drop_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop_p, is_causal=False
        )
        out = out.transpose(1, 2).reshape(B * L, L, C)
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
        
        # Pre-norm residual: x + attn(norm(x))
        # pair_mask is passed so padding positions are excluded from attention
        x_attn = x_attn + self._axial_attention(self.norm_row(x_attn), axis=1, pair_mask=pair_mask)
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

        outer = torch.einsum('bid,bje->bijde', a, b)          # (B,L,L,c,c)
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
        head_type: str = "cnn",
        head_num_heads: int = 4,  # Reduced default for efficiency
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
            reduction=fusion_reduction
        )
        
        # Input channels depend on fusion strategy output
        in_ch = self.fusion.out_channels
        
        # Input projection
        self.inp = nn.Conv2d(in_ch, width, 1)
        
        # Processing blocks - CNN, Dilated CNN, or Axial Attention
        if head_type == "axial":
            self.blocks = nn.ModuleList([
                AxialAttentionBlock(width, num_heads=head_num_heads) 
                for _ in range(depth)
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
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, pair_feat, prior, count, rel, esm_contacts, pair_mask=None):
        """
        Args:
            pair_feat: (B, d_pair, L, L) pairwise features
            prior: (B, 1, L, L) prior contact map (-1/0/1 or continuous BLOSUM)
            count: (B, 1, L, L) template count
            rel: (B, rel_ch, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            pair_mask: (B, 1, L, L) binary mask (1 = valid, 0 = padding)
            
        Returns:
            logits: (B, 1, L, L) contact prediction logits
        """
        # Apply fusion strategy
        x = self.fusion(pair_feat, prior, count, rel, esm_contacts)
        
        # Processing
        x = self.inp(x)
        if self.head_type == "axial":
            for block in self.blocks:
                if self.use_checkpoint and self.training:
                    x = grad_checkpoint(block, x, pair_mask, use_reentrant=False)
                else:
                    x = block(x, pair_mask=pair_mask)
        else:
            for block in self.blocks:
                x = block(x)
        logits = self.out(x)  # (B, 1, L, L)
        
        # Enforce symmetry
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits

import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models.components.fusion_strategies import (
    get_fusion_strategy,
    GroupedFeatureFusion,
    TruForFusion,
)
from src.models.components.triangle import TriangleMultiplicativeUpdate


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
        use_tpl_dist_bins: Whether to consume template distance-bin channels
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
        fusion_feature_groups: dict | None = None,
        use_template_features: bool = True,
        use_tpl_dist_bins: bool = False,
        head_type: str = "cnn",
        head_num_heads: int = 4,  # Reduced default for efficiency
        head_num_kv_heads: int = None,  # GQA: KV heads (None = same as head_num_heads)
        alternating_axial: bool = False,  # Alternating row/col instead of both per block
        use_depthwise: bool = False,
        use_checkpoint: bool = False,
        triangle_c: int = 32,
    ):
        super().__init__()

        self.head_type = head_type
        self.use_checkpoint = use_checkpoint
        self.use_template_features = bool(use_template_features)
        self.use_tpl_dist_bins = bool(use_tpl_dist_bins)

        tpl_dist_channels = 0
        if self.use_tpl_dist_bins:
            if not fusion_feature_groups:
                raise ValueError(
                    "use_tpl_dist_bins=True requires fusion_feature_groups "
                    "with a positive 'tpl_dist' channel count"
                )
            tpl_dist_channels = int(fusion_feature_groups.get("tpl_dist", 0))
            if tpl_dist_channels <= 0:
                raise ValueError(
                    "use_tpl_dist_bins=True requires a positive "
                    "fusion_feature_groups['tpl_dist'] channel count"
                )

        # Fusion strategy selection
        self.fusion_strategy = fusion_strategy
        self.fusion = get_fusion_strategy(
            strategy=fusion_strategy,
            d_pair=d_pair,
            d_rel=rel_ch,
            num_heads=fusion_num_heads,
            reduction=fusion_reduction,
            feature_groups=fusion_feature_groups,
            tpl_dist_channels=tpl_dist_channels,
        )
        self._frozen_template_bns: List[nn.Module] = []
        if not self.use_template_features:
            if isinstance(self.fusion, GroupedFeatureFusion):
                # Keep the checkpoint architecture compatible with the frontier,
                # but exclude absent-modality encoders from optimization and EMA.
                self.fusion.encoders.requires_grad_(False)
            elif isinstance(self.fusion, TruForFusion):
                # TruFor fuses two streams by cross-attention, so dropping the
                # template stream would change the fusion itself and answer a
                # different question. Keep the architecture and its parameters,
                # and feed constant zeros (see forward).
                #
                # That is only safe once the template encoder's BatchNorm is
                # pinned. A BN fed constant zeros collapses its running variance
                # (~1e-16 was measured on 2026-06-12) and then reports one thing
                # in train mode and another in eval; that invalidated the
                # b0cfw0w8 / kwm8qq4p controls. Fix the statistics at
                # mean 0 / var 1 and keep those modules in eval mode forever, so
                # normalisation is the identity and never updates. Affine
                # weights, convolutions and cross-attention stay trainable —
                # parameterisation is preserved, only the statistics are frozen.
                for m in self.fusion.template_encoder.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        with torch.no_grad():
                            m.running_mean.zero_()
                            m.running_var.fill_(1.0)
                            m.num_batches_tracked.zero_()
                        m.eval()
                        self._frozen_template_bns.append(m)
            else:
                raise ValueError(
                    "use_template_features=False is supported only with "
                    "fusion_strategy in {'grouped', 'trufor'}"
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
        elif head_type == "axial_tri":
            # Interleave axial blocks with triangle multiplicative updates:
            # [ax, ax, tri_out, tri_in, ax, ax, tri_out, tri_in, ...]
            blocks = []
            for i in range(depth):
                slot = i % 4
                if slot == 2:
                    blocks.append(
                        TriangleMultiplicativeUpdate(width, c_triangle=triangle_c, outgoing=True)
                    )
                elif slot == 3:
                    blocks.append(
                        TriangleMultiplicativeUpdate(width, c_triangle=triangle_c, outgoing=False)
                    )
                else:
                    blocks.append(
                        AxialAttentionBlock(
                            width,
                            num_heads=head_num_heads,
                            num_kv_heads=head_num_kv_heads,
                            axis=(1 + (i % 2)) if alternating_axial else 0,
                        )
                    )
            self.blocks = nn.ModuleList(blocks)
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
            raise ValueError(
                f"Unknown head_type: {head_type}. "
                "Must be 'cnn', 'dilated', 'axial', or 'axial_tri'"
            )
        
        # Output projection
        self.out = nn.Conv2d(width, 1, 1)
        # Prior probability init: contacts are ~5% of valid pairs.
        # bias = log(π/(1-π)) ≈ -2.94 so initial sigmoid ≈ 0.05.
        # This lets the model learn to push contacts UP from a low base,
        # rather than learning to suppress 95% of pairs from 0.5.
        # (RetinaNet, Lin et al. 2017)
        import math
        nn.init.constant_(self.out.bias, -math.log((1 - 0.05) / 0.05))

    def train(self, mode: bool = True):
        """Keep the empty-modality BatchNorm in eval mode across train()/eval().

        A single .eval() at construction is not enough: Lightning calls
        model.train() at the start of every training epoch, which would restart
        the running-statistics updates this control exists to prevent.
        """
        super().train(mode)
        for m in self._frozen_template_bns:
            m.eval()
        return self

    def forward(
        self,
        pair_feat,
        prior,
        count,
        rel,
        esm_contacts,
        pair_mask=None,
        tpl_dist_bins=None,
    ):
        """
        Args:
            pair_feat: (B, d_pair, L, L) pairwise features
            prior: (B, 1, L, L) prior contact map (-1/0/1 or continuous BLOSUM)
            count: (B, 1, L, L) template count
            rel: (B, rel_ch, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            pair_mask: (B, 1, L, L) binary mask (1 = valid, 0 = padding)
            tpl_dist_bins: optional Stage 2 per-group feature
                consumed when template distance channels are enabled.

        Returns:
            logits: (B, 1, L, L) contact prediction logits
        """
        # Apply fusion strategy — dispatch per-group features when using
        # GroupedFeatureFusion, else fall back to the legacy signature.
        if isinstance(self.fusion, GroupedFeatureFusion):
            feats: dict = {}
            if self.use_template_features:
                feats["tpl_contact"] = torch.cat([prior, count], dim=1)
                if tpl_dist_bins is not None:
                    feats["tpl_dist"] = tpl_dist_bins
            x = self.fusion(pair_feat, esm_contacts, rel, **feats)
        else:
            if not self.use_template_features:
                # Ignore whatever was passed and substitute constant zeros, so no
                # retrieval information can reach the model even if a caller
                # supplies real tensors. The distance bins must be MATERIALISED
                # here: with the prior builder disabled, collate_padded emits no
                # `tpl_dist_bins` at all, yet the encoder's input width is fixed
                # at 2 + tpl_dist_channels.
                b, _, h, w = pair_feat.shape
                z = functools.partial(
                    torch.zeros, dtype=pair_feat.dtype, device=pair_feat.device
                )
                prior = z((b, 1, h, w))
                count = z((b, 1, h, w))
                n_dist = getattr(self.fusion, "tpl_dist_channels", 0)
                tpl_dist_bins = z((b, n_dist, h, w)) if n_dist else None
            x = self.fusion(
                pair_feat,
                prior,
                count,
                rel,
                esm_contacts,
                tpl_dist_bins=tpl_dist_bins,
            )

        # Processing
        x = self.inp(x)
        if self.head_type == "axial":
            for block in self.blocks:
                if self.use_checkpoint and self.training:
                    x = grad_checkpoint(block, x, pair_mask, use_reentrant=False)
                else:
                    x = block(x, pair_mask=pair_mask)
        elif self.head_type == "axial_tri":
            # Triangle blocks run channel-last (B, L, L, C); axial blocks stay
            # channel-first (B, C, L, L) and handle their own permute.
            for block in self.blocks:
                if isinstance(block, TriangleMultiplicativeUpdate):
                    x = x.permute(0, 2, 3, 1).contiguous()
                    if self.use_checkpoint and self.training:
                        x = grad_checkpoint(block, x, pair_mask, use_reentrant=False)
                    else:
                        x = block(x, pair_mask)
                    x = x.permute(0, 3, 1, 2).contiguous()
                else:
                    if self.use_checkpoint and self.training:
                        x = grad_checkpoint(block, x, pair_mask, use_reentrant=False)
                    else:
                        x = block(x, pair_mask=pair_mask)
        else:
            for block in self.blocks:
                x = block(x)
        logits = self.out(x)  # (B, 1, L, L)
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        return logits

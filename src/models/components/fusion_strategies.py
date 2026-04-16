"""
Feature fusion strategies for contact prediction.

Provides multiple fusion approaches:
- StandardFusion: Gated residual + auxiliary concatenation (current baseline)
- TruForFusion: Cross-modal attention-based fusion inspired by TruFor architecture
"""

import torch
import torch.nn as nn
from typing import Tuple


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism from TruFor.
    
    Each stream queries the context from the other stream:
    - ctx1 = (k1.T @ v1).softmax()  # Context from stream 1
    - ctx2 = (k2.T @ v2).softmax()  # Context from stream 2
    - out1 = q1 @ ctx2              # Stream 1 attends to stream 2's context
    - out2 = q2 @ ctx1              # Stream 2 attends to stream 1's context
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate K,V projections for each stream
        self.kv1 = nn.Linear(dim, dim * 2, bias=False)
        self.kv2 = nn.Linear(dim, dim * 2, bias=False)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: (B, N, C) features from stream 1
            x2: (B, N, C) features from stream 2
            
        Returns:
            Tuple of (out1, out2) with cross-attended features
        """
        B, N, C = x1.shape
        
        # Use input directly as queries (no projection in TruFor)
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        # Project to keys and values
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        
        # bf16 has same exponent range as fp32, no overflow risk — skip upcast
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        
        # Cross-attend: stream 1's queries attend to stream 2's context
        x1_out = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2_out = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        
        return x1_out, x2_out


class CrossPath(nn.Module):
    """
    Cross-path fusion with bidirectional cross-attention.
    
    Architecture:
    1. Split each stream into (y, u) via Linear projection + ReLU
    2. Cross-attend on u: (v1, v2) = CrossAttention(u1, u2)
    3. Concatenate: y1 + v1, y2 + v2
    4. Project back and add residual
    
    Args:
        dim: Feature dimension
        reduction: Channel reduction factor (default=1, no reduction)
        num_heads: Number of attention heads
    """
    def __init__(self, dim: int, reduction: int = 1, num_heads: int = 8):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: (B, N, C) features from stream 1
            x2: (B, N, C) features from stream 2
            
        Returns:
            Tuple of (out1, out2) with fused features
        """
        # Split into (y, u) for each stream
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        
        # Cross-attend on u
        v1, v2 = self.cross_attn(u1, u2)
        
        # Concatenate and project back
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        
        # Residual connection + normalization
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    """
    Channel embedding for fused features.
    
    Args:
        in_channels: Input channels (usually dim*2 after concatenation)
        out_channels: Output channels
        reduction: Channel reduction factor
        norm_layer: Normalization layer type
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        reduction: int = 1, 
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels // reduction, 1, bias=False)
        self.norm1 = norm_layer(out_channels // reduction)
        self.act = nn.ReLU()
        self.conv_expand = nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False)
        # Gate projection: derives attention from the skip (pre-bottleneck)
        # input and projects to out_channels so it can modulate x_expanded.
        self.gate_proj = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.gate = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) flattened features
            H: Height for reshaping
            W: Width for reshaping
            
        Returns:
            (B, out_channels, H, W) embedded features
        """
        B, N, C = x.shape
        # Reshape to 2D: (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        
        # Channel attention with cross-gating:
        # gate is derived from the original (skip) input via gate_proj,
        # applied to the bottleneck-expanded features.
        x_reduced = self.act(self.norm1(self.conv_reduce(x)))
        x_expanded = self.conv_expand(x_reduced)
        
        # Gated output: gate from skip-connection, value from bottleneck
        out = x_expanded * self.gate(self.gate_proj(x))
        
        return out


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module (FFM) from TruFor.
    
    Combines two streams via:
    1. CrossPath: Bidirectional cross-attention
    2. Concatenation: Cat(x1, x2)
    3. ChannelEmbed: Channel attention and projection
    
    Args:
        dim: Feature dimension
        reduction: Channel reduction factor
        num_heads: Number of attention heads
        norm_layer: Normalization layer type
    """
    def __init__(
        self, 
        dim: int, 
        reduction: int = 1, 
        num_heads: int = 8, 
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2, 
            out_channels=dim, 
            reduction=reduction, 
            norm_layer=norm_layer
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: (B, C, H, W) features from stream 1
            x2: (B, C, H, W) features from stream 2
            
        Returns:
            (B, C, H, W) fused features
        """
        B, C, H, W = x1.shape
        
        # Flatten to sequence: (B, C, H, W) -> (B, H*W, C)
        x1_seq = x1.flatten(2).transpose(1, 2)
        x2_seq = x2.flatten(2).transpose(1, 2)
        
        # Cross-path fusion
        x1_fused, x2_fused = self.cross(x1_seq, x2_seq)
        
        # Concatenate and embed
        merge = torch.cat((x1_fused, x2_fused), dim=-1)  # (B, H*W, 2C)
        out = self.channel_emb(merge, H, W)  # (B, C, H, W)
        
        return out


class TemplateEmbedder(nn.Module):
    """Per-group 1×1 encoders with learned gates for heterogeneous template channels.

    Each channel group (struct, dist, ss) gets its own small MLP that maps raw
    features into a shared *d*-dimensional space.  A per-channel sigmoid gate
    controls how much each group contributes before they are summed, so the
    model can learn to down-weight noisy or uninformative groups.

    Args:
        d_out: Output embedding dimension (should match d_pair).
        n_dist_bins: Number of template distance-bin channels (0 = disabled).
        n_ss_feat: Number of template SS-pair feature channels (0 = disabled).
    """

    def __init__(self, d_out: int, n_dist_bins: int = 0, n_ss_feat: int = 0):
        super().__init__()
        self.d_out = d_out

        # Group 1 – always present: prior (1) + count (1)
        self.mlp_struct = nn.Sequential(
            nn.Conv2d(2, d_out, 1), nn.ReLU(), nn.Conv2d(d_out, d_out, 1),
        )
        self.gate_struct = nn.Parameter(torch.zeros(1, d_out, 1, 1))

        # Group 2 – distance bins (optional)
        self.has_dist = n_dist_bins > 0
        if self.has_dist:
            self.mlp_dist = nn.Sequential(
                nn.Conv2d(n_dist_bins, d_out, 1), nn.ReLU(), nn.Conv2d(d_out, d_out, 1),
            )
            self.gate_dist = nn.Parameter(torch.zeros(1, d_out, 1, 1))

        # Group 3 – SS-pair features (optional)
        self.has_ss = n_ss_feat > 0
        if self.has_ss:
            self.mlp_ss = nn.Sequential(
                nn.Conv2d(n_ss_feat, d_out, 1), nn.ReLU(), nn.Conv2d(d_out, d_out, 1),
            )
            self.gate_ss = nn.Parameter(torch.zeros(1, d_out, 1, 1))

    def forward(
        self,
        prior: torch.Tensor,
        count: torch.Tensor,
        dist_bins: torch.Tensor = None,
        ss_feat: torch.Tensor = None,
        return_intermediates: bool = False,
    ):
        """
        Returns:
            (B, d_out, L, L) gated-sum template embedding.
            If return_intermediates, also returns dict with per-group stats.
        """
        struct_in = torch.cat([prior, count / 4.0], dim=1)  # (B, 2, L, L)  count normalised to [0,1]
        h_struct = self.mlp_struct(struct_in)
        g_struct = torch.sigmoid(self.gate_struct)
        h = g_struct * h_struct

        h_dist = g_dist = None
        if self.has_dist and dist_bins is not None:
            h_dist = self.mlp_dist(dist_bins)
            g_dist = torch.sigmoid(self.gate_dist)
            h = h + g_dist * h_dist

        h_ss = g_ss = None
        if self.has_ss and ss_feat is not None:
            h_ss = self.mlp_ss(ss_feat)
            g_ss = torch.sigmoid(self.gate_ss)
            h = h + g_ss * h_ss

        if not return_intermediates:
            return h

        diag = {
            "gate_struct": g_struct.mean().item(),
            "gate_dist": g_dist.mean().item() if g_dist is not None else 0.0,
            "gate_ss": g_ss.mean().item() if g_ss is not None else 0.0,
            "h_struct_norm": h_struct.norm().item(),
            "h_dist_norm": h_dist.norm().item() if h_dist is not None else 0.0,
            "h_ss_norm": h_ss.norm().item() if h_ss is not None else 0.0,
            "tpl_emb_mean": h.mean().item(),
            "tpl_emb_std": h.std().item(),
        }
        return h, diag


class StandardFusion(nn.Module):
    """
    Standard fusion strategy (current baseline).
    
    Architecture:
    1. TemplateEmbedder: per-group MLPs + gated sum → d_pair embedding
    2. Gated residual: gate = sigmoid(conv(tpl_emb)); x = esm + gate * tpl_emb
    3. Aux concat: x = cat([x, tpl_emb, rel_proj(rel)])
    
    Args:
        d_pair: Dimension of pairwise features
        d_rel: Dimension of relative position embeddings
        n_dist_bins: Number of template distance bin channels (0 = no distance bins)
        n_ss_feat: Number of template SS-pair feature channels (0 = disabled)
    """
    def __init__(self, d_pair: int, d_rel: int, n_dist_bins: int = 0, n_ss_feat: int = 0):
        super().__init__()
        self.n_dist_bins = n_dist_bins
        self.n_ss_feat = n_ss_feat
        
        # ESM semantic encoder: combines pair_feat + optional esm_contacts
        esm_in_channels = d_pair + 1
        self.esm_proj = nn.Conv2d(esm_in_channels, d_pair, 1)
        
        # Template embedder: per-group MLPs + gated sum → d_pair
        self.tpl_embed = TemplateEmbedder(d_out=d_pair, n_dist_bins=n_dist_bins, n_ss_feat=n_ss_feat)
        
        # Gate network: learns spatial confidence from template embedding
        self.gate_conv = nn.Conv2d(d_pair, 1, 1)
        
        # Relative position projection
        self.rel_proj = nn.Conv2d(d_rel, d_rel, 1)
        
        # Total input channels: d_pair (gated residual) + d_pair (tpl_emb) + d_rel
        self.out_channels = d_pair + d_pair + d_rel
        
    def forward(
        self, 
        pair_feat: torch.Tensor,
        prior: torch.Tensor,
        count: torch.Tensor,
        rel: torch.Tensor,
        esm_contacts: torch.Tensor,
        dist_bins: torch.Tensor = None,
        ss_feat: torch.Tensor = None,
        return_intermediates: bool = False,
    ):
        # Build ESM semantic stream (learned from sequences)
        esm_semantic = torch.cat([pair_feat, esm_contacts], dim=1)
        esm_semantic = self.esm_proj(esm_semantic)
        
        # Template embedding via per-group gated sum
        tpl_out = self.tpl_embed(prior, count, dist_bins, ss_feat,
                                 return_intermediates=return_intermediates)
        if return_intermediates:
            tpl_emb, tpl_diag = tpl_out
        else:
            tpl_emb = tpl_out
        
        # Compute spatial gate from template embedding
        gate = torch.sigmoid(self.gate_conv(tpl_emb))
        x = esm_semantic + gate * tpl_emb
        
        rel_emb = self.rel_proj(rel)
        x_fused = torch.cat([x, tpl_emb, rel_emb], dim=1)
        
        if not return_intermediates:
            return x_fused

        diag = tpl_diag
        diag["esm_feat_norm"] = esm_semantic.norm().item()
        diag["tpl_feat_norm"] = tpl_emb.norm().item()
        diag["stream_ratio"] = tpl_emb.norm().item() / (esm_semantic.norm().item() + 1e-8)
        diag["fusion_gate_mean"] = gate.mean().item()
        diag["x_fused_mean"] = x_fused.mean().item()
        diag["x_fused_std"] = x_fused.std().item()
        return x_fused, diag


class TruForFusion(nn.Module):
    """
    TruFor-inspired cross-modal fusion strategy.
    
    Architecture (Option A - Clean TruFor Analogy):
    - Stream 1 (Semantic): ESM2 pair_feat + esm_contacts (both learned from sequences)
    - Stream 2 (Fingerprint): Template prior + count (both from 3D structures)
    - Cross-attention fusion between the two streams
    - Add relative position as auxiliary
    
    This mirrors TruFor's design:
    - TruFor Stream 1: RGB (image content)
    - TruFor Stream 2: Noiseprint++ (sensor noise)
    - Our Stream 1: ESM2 features (sequence patterns)
    - Our Stream 2: Template features (structural evidence)
    
    Args:
        d_pair: Dimension of pairwise features
        d_rel: Dimension of relative position embeddings
        num_heads: Number of attention heads for cross-attention
        reduction: Channel reduction factor in fusion
    """
    def __init__(
        self, 
        d_pair: int, 
        d_rel: int, 
        num_heads: int = 8, 
        reduction: int = 1,
        n_dist_bins: int = 0,
        n_ss_feat: int = 0,
    ):
        super().__init__()
        self.n_dist_bins = n_dist_bins
        self.n_ss_feat = n_ss_feat
        
        # ESM semantic stream encoder: [pair_feat, optional esm_contacts] -> d_pair
        esm_in_channels = d_pair + 1
        self.esm_encoder = nn.Sequential(
            nn.Conv2d(esm_in_channels, d_pair, 3, padding=1),
            nn.BatchNorm2d(d_pair),
            nn.ReLU()
        )
        
        # Template embedder: per-group MLPs + gated sum → d_pair
        self.tpl_embed = TemplateEmbedder(d_out=d_pair, n_dist_bins=n_dist_bins, n_ss_feat=n_ss_feat)
        
        # Cross-modal fusion
        self.ffm = FeatureFusionModule(
            dim=d_pair, 
            reduction=reduction, 
            num_heads=num_heads,
        )
        
        # Relative position projection
        self.rel_proj = nn.Conv2d(d_rel, d_rel, 1)
        
        # Total output: fused + rel = d_pair + d_rel
        self.out_channels = d_pair + d_rel
        
    def forward(
        self, 
        pair_feat: torch.Tensor,
        prior: torch.Tensor,
        count: torch.Tensor,
        rel: torch.Tensor,
        esm_contacts: torch.Tensor,
        dist_bins: torch.Tensor = None,
        ss_feat: torch.Tensor = None,
        return_intermediates: bool = False,
    ):
        # Build Stream 1: ESM semantic (learned from sequences)
        esm_input = torch.cat([pair_feat, esm_contacts], dim=1)
        esm_feat = self.esm_encoder(esm_input)
        
        # Build Stream 2: Template embedding via per-group gated sum
        tpl_out = self.tpl_embed(prior, count, dist_bins, ss_feat,
                                 return_intermediates=return_intermediates)
        if return_intermediates:
            template_feat, tpl_diag = tpl_out
        else:
            template_feat = tpl_out
        
        # Cross-modal fusion: ESM semantic <-> Template fingerprint
        x_fused = self.ffm(esm_feat, template_feat)
        
        # Add relative position information
        rel_emb = self.rel_proj(rel)
        x_final = torch.cat([x_fused, rel_emb], dim=1)
        
        if not return_intermediates:
            return x_final

        diag = tpl_diag
        diag["esm_feat_norm"] = esm_feat.norm().item()
        diag["tpl_feat_norm"] = template_feat.norm().item()
        diag["stream_ratio"] = template_feat.norm().item() / (esm_feat.norm().item() + 1e-8)
        diag["x_fused_mean"] = x_final.mean().item()
        diag["x_fused_std"] = x_final.std().item()
        return x_final, diag


def get_fusion_strategy(
    strategy: str, 
    d_pair: int, 
    d_rel: int, 
    num_heads: int = 8, 
    reduction: int = 1,
    n_dist_bins: int = 0,
    n_ss_feat: int = 0,
) -> nn.Module:
    """
    Factory function to create fusion strategy.
    
    Args:
        strategy: "standard" or "trufor"
        d_pair: Dimension of pairwise features
        d_rel: Dimension of relative position embeddings
        num_heads: Number of attention heads (TruFor only)
        reduction: Channel reduction factor (TruFor only)
        n_dist_bins: Number of template distance bin channels
        n_ss_feat: Number of template SS-pair feature channels
        
    Returns:
        Fusion module instance
    """
    if strategy == "standard":
        return StandardFusion(d_pair=d_pair, d_rel=d_rel, n_dist_bins=n_dist_bins, n_ss_feat=n_ss_feat)
    elif strategy == "trufor":
        return TruForFusion(
            d_pair=d_pair, 
            d_rel=d_rel, 
            num_heads=num_heads, 
            reduction=reduction,
            n_dist_bins=n_dist_bins,
            n_ss_feat=n_ss_feat,
        )
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}. Choose 'standard' or 'trufor'.")

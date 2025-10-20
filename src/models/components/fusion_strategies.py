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
        
        # Compute context: (B, H, C//H, C//H)
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale  # Context from stream 1
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale  # Context from stream 2
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
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
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
        self.act = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False)
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
        
        # Channel attention
        x_reduced = self.act(self.norm1(self.conv_reduce(x)))
        x_expanded = self.conv_expand(x_reduced)
        
        # Gated output
        out = x_expanded * self.gate(x_expanded)
        
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


class StandardFusion(nn.Module):
    """
    Standard fusion strategy (current baseline).
    
    Architecture:
    1. Gated prior: gate = sigmoid(conv([prior, count]))
    2. Residual: x = esm_semantic + gate * prior_phi(prior)
    3. Aux concat: x = cat([x, prior, count, rel_proj(rel)])
    
    Args:
        d_pair: Dimension of pairwise features
        d_rel: Dimension of relative position embeddings
    """
    def __init__(self, d_pair: int, d_rel: int):
        super().__init__()
        
        # ESM semantic encoder: combines pair_feat + optional esm_contacts
        esm_in_channels = d_pair + 1
        self.esm_proj = nn.Conv2d(esm_in_channels, d_pair, 1)
        
        # Gate network: learns confidence from prior and count
        self.gate_conv = nn.Conv2d(2, 1, 1)  # [prior, count] -> gate
        
        # Prior embedding: maps prior values to feature space
        self.prior_phi = nn.Conv2d(1, d_pair, 1)
        
        # Relative position projection
        self.rel_proj = nn.Conv2d(d_rel, d_rel, 1)
        
        # Total input channels: pair_feat + prior + count + rel = d_pair + 1 + 1 + d_rel
        self.out_channels = d_pair + 1 + 1 + d_rel
        
    def forward(
        self, 
        pair_feat: torch.Tensor,
        prior: torch.Tensor,
        count: torch.Tensor,
        rel: torch.Tensor,
        esm_contacts: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pair_feat: (B, d_pair, L, L) pairwise features from PairFeatures
            prior: (B, 1, L, L) prior contact map (-1/0/1 or continuous BLOSUM scores)
            count: (B, 1, L, L) template count confidence
            rel: (B, d_rel, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            
        Returns:
            (B, out_channels, L, L) fused features
        """
        # Build ESM semantic stream (learned from sequences)
        esm_semantic = torch.cat([pair_feat, esm_contacts], dim=1)  # (B, d_pair+1, L, L)
        esm_semantic = self.esm_proj(esm_semantic)  # (B, d_pair, L, L)
        
        # Compute gate from prior and count
        gate_input = torch.cat([prior, count], dim=1)  # (B, 2, L, L)
        gate = torch.sigmoid(self.gate_conv(gate_input))  # (B, 1, L, L)
        
        # Gated residual: add prior information modulated by confidence
        prior_emb = self.prior_phi(prior)  # (B, d_pair, L, L)
        x = esm_semantic + gate * prior_emb  # (B, d_pair, L, L)
        
        # Auxiliary features: concatenate raw information
        rel_emb = self.rel_proj(rel)  # (B, d_rel, L, L)
        aux = [prior, count, rel_emb]  # Raw template info + relative position
        
        # Final concatenation
        x_fused = torch.cat([x] + aux, dim=1)  # (B, d_pair+1+1+d_rel, L, L)
        
        return x_fused


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
        reduction: int = 1
    ):
        super().__init__()
        # ESM semantic stream encoder: [pair_feat, optional esm_contacts] -> d_pair
        esm_in_channels = d_pair + 1
        self.esm_encoder = nn.Sequential(
            nn.Conv2d(esm_in_channels, d_pair, 3, padding=1),
            nn.BatchNorm2d(d_pair),
            nn.ReLU(inplace=True)
        )
        
        # Template fingerprint encoder: [prior, count] -> d_pair dimensions
        self.template_encoder = nn.Sequential(
            nn.Conv2d(2, d_pair // 2, 3, padding=1),
            nn.BatchNorm2d(d_pair // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_pair // 2, d_pair, 3, padding=1),
            nn.BatchNorm2d(d_pair),
            nn.ReLU(inplace=True)
        )
        
        # Cross-modal fusion
        self.ffm = FeatureFusionModule(
            dim=d_pair, 
            reduction=reduction, 
            num_heads=num_heads,
            norm_layer=nn.BatchNorm2d
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
        esm_contacts: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pair_feat: (B, d_pair, L, L) pairwise features from ESM2 (semantic)
            prior: (B, 1, L, L) prior contact map from templates (fingerprint)
            count: (B, 1, L, L) template count (confidence)
            rel: (B, d_rel, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            
        Returns:
            (B, out_channels, L, L) cross-fused features
        """
        # Build Stream 1: ESM semantic (learned from sequences)
        esm_input = torch.cat([pair_feat, esm_contacts], dim=1)  # (B, d_pair+1, L, L)
        esm_feat = self.esm_encoder(esm_input)  # (B, d_pair, L, L)
        
        # Build Stream 2: Template fingerprint (measured from structures)
        template_input = torch.cat([prior, count], dim=1)  # (B, 2, L, L)
        template_feat = self.template_encoder(template_input)  # (B, d_pair, L, L)
        
        # Cross-modal fusion: ESM semantic <-> Template fingerprint
        # FFM uses cross-attention to let each stream attend to the other
        x_fused = self.ffm(esm_feat, template_feat)  # (B, d_pair, L, L)
        
        # Add relative position information
        rel_emb = self.rel_proj(rel)  # (B, d_rel, L, L)
        x_final = torch.cat([x_fused, rel_emb], dim=1)  # (B, d_pair+d_rel, L, L)
        
        return x_final


def get_fusion_strategy(
    strategy: str, 
    d_pair: int, 
    d_rel: int, 
    num_heads: int = 8, 
    reduction: int = 1
) -> nn.Module:
    """
    Factory function to create fusion strategy.
    
    Args:
        strategy: "standard" or "trufor"
        d_pair: Dimension of pairwise features
        d_rel: Dimension of relative position embeddings
        num_heads: Number of attention heads (TruFor only)
        reduction: Channel reduction factor (TruFor only)
        
    Returns:
        Fusion module instance
    """
    if strategy == "standard":
        return StandardFusion(d_pair=d_pair, d_rel=d_rel)
    elif strategy == "trufor":
        return TruForFusion(
            d_pair=d_pair, 
            d_rel=d_rel, 
            num_heads=num_heads, 
            reduction=reduction
        )
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}. Choose 'standard' or 'trufor'.")

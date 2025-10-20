import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.fusion_strategies import get_fusion_strategy

_relpos_cache = {}


def relpos_buckets(L, device, cuts=(0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64)):
    """
    Returns a relative-position encoding: (R, L, L), where R = len(cuts)+1.
    Bin k means |i-j| ∈ (cuts[k-1], cuts[k]] with k=0 for |i-j| <= cuts[0].
    """
    key = (L, device, tuple(cuts))
    if key in _relpos_cache:
        return _relpos_cache[key]

    idx = torch.arange(L, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()  # (L, L)
    edges = torch.tensor(cuts, device=device)
    bins = torch.bucketize(dist, edges)  # (L, L) in [0..len(cuts)]
    R = len(cuts) + 1
    oh = F.one_hot(bins.clamp_max(R - 1), num_classes=R)  # (L, L, R)
    rel = oh.permute(2, 0, 1).float()  # (R, L, L)

    _relpos_cache[key] = rel
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
    Memory-efficient axial attention block for 2D pairwise features.
    
    Optimizations:
    - Shared QKV projections for row and column attention
    - Reduced FFN expansion (2x instead of 4x)
    - Optional checkpoint for memory savings
    - Fused operations where possible
    
    Args:
        channels: Number of channels
        num_heads: Number of attention heads (must divide channels)
        dropout: Dropout rate
        ffn_expansion: FFN expansion factor (default: 2, standard is 4)
    """
    def __init__(
        self, 
        channels: int, 
        num_heads: int = 4,  # Reduced from 8 for efficiency
        dropout: float = 0.1,
        ffn_expansion: int = 2  # Reduced from 4 for memory
    ):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Shared projections for both row and column attention
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Lightweight FFN (2x expansion instead of 4x)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * ffn_expansion, channels),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _efficient_attention(self, x, axis):
        """
        Memory-efficient attention along one axis.
        
        Args:
            x: (B, L, L, C) tensor
            axis: 1 for row attention, 2 for column attention
        """
        B, L, _, C = x.shape
        
        # Rearrange based on axis
        if axis == 1:  # Row attention: attend across columns for each row
            x_seq = x.reshape(B * L, L, C)  # (B*L, L, C) - each row is a sequence
        else:  # Column attention: attend across rows for each column
            x_seq = x.transpose(1, 2).reshape(B * L, L, C)  # (B*L, L, C) - each col is a sequence
        
        # Compute QKV
        qkv = self.qkv(x_seq).reshape(B * L, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*L, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B*L, H, L, D)
        
        # Efficient attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*L, H, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (B*L, H, L, D)
        out = out.transpose(1, 2).reshape(B * L, L, C)  # (B*L, L, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        # Reshape back
        if axis == 1:
            out = out.reshape(B, L, L, C)
        else:
            out = out.reshape(B, L, L, C).transpose(1, 2)
        
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L, L) 2D pairwise features
            
        Returns:
            (B, C, L, L) attended features
        """
        B, C, L, _ = x.shape
        
        # Convert to (B, L, L, C) for attention operations
        x_attn = x.permute(0, 2, 3, 1)  # (B, L, L, C)
        
        # Row attention
        x_attn = self.norm1(x_attn + self._efficient_attention(x_attn, axis=1))
        
        # Column attention
        x_attn = self.norm1(x_attn + self._efficient_attention(x_attn, axis=2))
        
        # FFN
        x_attn = self.norm2(x_attn + self.ffn(x_attn))
        
        # Back to (B, C, L, L)
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
        use_depthwise: bool = False
    ):
        super().__init__()
        
        self.head_type = head_type
        
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
            self.blocks = nn.Sequential(*[
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

    def forward(self, pair_feat, prior, count, rel, esm_contacts):
        """
        Args:
            pair_feat: (B, d_pair, L, L) pairwise features
            prior: (B, 1, L, L) prior contact map (-1/0/1 or continuous BLOSUM)
            count: (B, 1, L, L) template count
            rel: (B, rel_ch, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            
        Returns:
            logits: (B, 1, L, L) contact prediction logits
        """
        # Apply fusion strategy
        x = self.fusion(pair_feat, prior, count, rel, esm_contacts)
        
        # CNN processing
        x = self.inp(x)
        x = self.blocks(x)
        logits = self.out(x)  # (B, 1, L, L)
        
        # Enforce symmetry
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits

import torch
from src.models.components.pair_features import PairFeatures
from src.models.components.pair2d_head import Pair2DHead


class ContactModel(torch.nn.Module):
    """
    Contact prediction model with ESM2 backbone and configurable fusion.
    
    Args:
        d_esm: ESM2 embedding dimension
        d_pair: Pairwise feature dimension
        width: Hidden dimension for Pair2DHead
        depth: Number of processing blocks in Pair2DHead
        rel_ch: Relative position embedding dimension
        fusion_strategy: "standard" or "trufor"
        fusion_num_heads: Number of attention heads for TruFor
        fusion_reduction: Channel reduction for TruFor
        head_type: "cnn", "dilated" or "axial" - architecture type for head
        head_num_heads: Number of attention heads if head_type="axial"
        use_depthwise: Whether to use depthwise separable convs if head_type="cnn"
    """
    def __init__(
        self, 
        d_esm: int = 1280, 
        d_pair: int = 128, 
        width: int = 128, 
        depth: int = 8, 
        rel_ch: int = 14,
        fusion_strategy: str = "standard",
        fusion_num_heads: int = 8,
        fusion_reduction: int = 1,
        head_type: str = "cnn",
        head_num_heads: int = 8,
        use_depthwise: bool = False
    ):
        super().__init__()
        self.pair = PairFeatures(d_model=d_esm, d_pair=d_pair)
        
        self.head = Pair2DHead(
            d_pair=d_pair, 
            width=width, 
            depth=depth, 
            rel_ch=rel_ch,
            fusion_strategy=fusion_strategy,
            fusion_num_heads=fusion_num_heads,
            fusion_reduction=fusion_reduction,
            head_type=head_type,
            head_num_heads=head_num_heads,
            use_depthwise=use_depthwise
        )

    def forward(self, h_esm, prior, count, rel, esm_contacts=None):
        """
        Args:
            h_esm: (B, L, d_esm) ESM2 embeddings
            prior: (B, 1, L, L) prior contact map from templates
            count: (B, 1, L, L) template count
            rel: (B, rel_ch, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            
        Returns:
            logits: (B, 1, L, L) contact prediction logits
        """
        # Generate pairwise features from ESM2 embeddings
        pair_feat = self.pair(h_esm)  # (B, d_pair, L, L)
        
        # Pass to fusion head (esm_contacts integrated in Stream 1 if use_esm_contacts=True)
        logits = self.head(pair_feat, prior, count, rel=rel, esm_contacts=esm_contacts)
        return logits

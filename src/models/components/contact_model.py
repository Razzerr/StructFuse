import torch
import torch.nn as nn
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
        late_fusion: If True, skip template features in early fusion and add
                     learned α·prior bias directly to logits.
    """
    def __init__(
        self, 
        d_esm: int = 1280, 
        d_pair: int = 128, 
        width: int = 128, 
        depth: int = 8, 
        rel_ch: int = 14,
        rank: int = 32,
        fusion_strategy: str = "standard",
        fusion_num_heads: int = 8,
        fusion_reduction: int = 1,
        n_dist_bins: int = 0,
        n_ss_feat: int = 0,
        n_out: int = 1,  # 1 for binary contact, N for distogram
        head_type: str = "cnn",
        head_num_heads: int = 8,
        head_num_kv_heads: int = None,
        alternating_axial: bool = False,
        use_depthwise: bool = False,
        use_checkpoint: bool = False,
        late_fusion: bool = False,
    ):
        super().__init__()
        self.late_fusion = late_fusion
        self.pair = PairFeatures(d_model=d_esm, d_pair=d_pair, rank=rank)
        
        self.head = Pair2DHead(
            d_pair=d_pair, 
            width=width, 
            depth=depth, 
            rel_ch=rel_ch,
            fusion_strategy=fusion_strategy,
            fusion_num_heads=fusion_num_heads,
            fusion_reduction=fusion_reduction,
            # Late fusion: head sees no template features at all
            n_dist_bins=0 if late_fusion else n_dist_bins,
            n_ss_feat=0 if late_fusion else n_ss_feat,
            n_out=n_out,
            head_type=head_type,
            head_num_heads=head_num_heads,
            head_num_kv_heads=head_num_kv_heads,
            alternating_axial=alternating_axial,
            use_depthwise=use_depthwise,
            use_checkpoint=use_checkpoint,
        )

        if late_fusion:
            # Learned scalar bias: logits += alpha * prior
            # Init to 0 → model starts identical to no-template baseline
            self.late_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h_esm, prior, count, rel, esm_contacts=None, pair_mask=None, dist_bins=None, ss_feat=None, return_intermediates=False):
        # Generate pairwise features from ESM2 embeddings
        pair_feat = self.pair(h_esm)  # (B, d_pair, L, L)
        
        # Late fusion: head only sees prior+count (no dist_bins/ss_feat)
        head_dist_bins = None if self.late_fusion else dist_bins
        head_ss_feat = None if self.late_fusion else ss_feat

        head_out = self.head(
            pair_feat, prior, count, rel=rel, esm_contacts=esm_contacts,
            pair_mask=pair_mask, dist_bins=head_dist_bins, ss_feat=head_ss_feat,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            logits, diag = head_out
            diag["pair_feat_mean"] = pair_feat.mean().item()
            diag["pair_feat_std"] = pair_feat.std().item()
        else:
            logits = head_out
            diag = None

        # Late fusion: add learned α·prior directly to logits
        if self.late_fusion:
            logits = logits + self.late_alpha * prior
            if diag is not None:
                diag["late_alpha"] = self.late_alpha.item()

        if diag is not None:
            return logits, diag
        return logits

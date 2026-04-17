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
        fusion_strategy: "standard", "trufor", or "grouped"
        fusion_num_heads: Number of attention heads for TruFor
        fusion_reduction: Channel reduction for TruFor
        fusion_feature_groups: dict[str, int] for "grouped" fusion
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
        rank: int = 32,
        fusion_strategy: str = "standard",
        fusion_num_heads: int = 8,
        fusion_reduction: int = 1,
        fusion_feature_groups: dict | None = None,
        head_type: str = "cnn",
        head_num_heads: int = 8,
        head_num_kv_heads: int = None,
        alternating_axial: bool = False,
        use_depthwise: bool = False,
        use_checkpoint: bool = False,
        triangle_c: int = 32,
        use_distogram_head: bool = False,
        n_distogram_bins: int = 8,
    ):
        super().__init__()
        self.pair = PairFeatures(d_model=d_esm, d_pair=d_pair, rank=rank)

        self.head = Pair2DHead(
            d_pair=d_pair,
            width=width,
            depth=depth,
            rel_ch=rel_ch,
            fusion_strategy=fusion_strategy,
            fusion_num_heads=fusion_num_heads,
            fusion_reduction=fusion_reduction,
            fusion_feature_groups=fusion_feature_groups,
            head_type=head_type,
            head_num_heads=head_num_heads,
            head_num_kv_heads=head_num_kv_heads,
            alternating_axial=alternating_axial,
            use_depthwise=use_depthwise,
            use_checkpoint=use_checkpoint,
            triangle_c=triangle_c,
            use_distogram_head=use_distogram_head,
            n_distogram_bins=n_distogram_bins,
        )

    def forward(
        self,
        h_esm,
        prior,
        count,
        rel,
        esm_contacts=None,
        pair_mask=None,
        tpl_dist_bins=None,
        tpl_agreement=None,
        tpl_dist_stats=None,
    ):
        """
        Args:
            h_esm: (B, L, d_esm) ESM2 embeddings
            prior: (B, 1, L, L) prior contact map from templates
            count: (B, 1, L, L) template count
            rel: (B, rel_ch, L, L) relative position embeddings
            esm_contacts: (B, 1, L, L) ESM2 contact predictions
            pair_mask: (B, 1, L, L) binary mask (1 = valid, 0 = padding)
            tpl_dist_bins: (B, 9, L, L) per-pair soft distance bin histograms
                from aggregated templates (Stage 2 feature). Optional.
            tpl_agreement: (B, 1, L, L) 1 - std across top-K template
                projected contacts (Stage 2 feature). Optional.
            tpl_dist_stats: (B, 2, L, L) mean/std of projected Cα-Cα
                distances across templates (Stage 2 feature). Optional.

        Returns:
            logits: (B, 1, L, L) contact prediction logits
        """
        pair_feat = self.pair(h_esm)  # (B, d_pair, L, L)

        logits = self.head(
            pair_feat,
            prior,
            count,
            rel=rel,
            esm_contacts=esm_contacts,
            pair_mask=pair_mask,
            tpl_dist_bins=tpl_dist_bins,
            tpl_agreement=tpl_agreement,
            tpl_dist_stats=tpl_dist_stats,
        )
        return logits

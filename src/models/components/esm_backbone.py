import torch
from src.models.components.esm import pretrained


class ESM2Backbone(torch.nn.Module):
    """
    ESM2 protein language model backbone with optional contact prediction.

    No caching — with cluster sampling each epoch sees different chains,
    so caching yields ~0% hit rate and leaks memory.

    Args:
        model_name: ESM2 model variant (e.g., "esm2_t33_650M_UR50D")
        finetune: Whether to fine-tune ESM2 parameters
    """
    def __init__(
        self, 
        model_name: str = "esm2_t33_650M_UR50D", 
        finetune: bool = False
    ):
        super().__init__()
        self.model, self.alphabet = getattr(pretrained, model_name)()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.finetune = finetune
        
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, seq_list, device):
        """Run ESM2 forward pass.

        Args:
            seq_list: list of (name, sequence) tuples
            device: torch device

        Returns:
            rep: (B, L, D) per-residue representations
            contacts: (B, 1, L, L) contact probabilities from attention
        """
        _, _, tokens = self.batch_converter(seq_list)
        tokens = tokens.to(device)

        ctx = torch.no_grad() if not self.finetune else torch.enable_grad()
        with ctx:
            out = self.model(
                tokens, repr_layers=[self.model.num_layers], return_contacts=True
            )
            rep = out["representations"][self.model.num_layers][:, 1:-1, :]
            # ContactPredictionHead already strips BOS/EOS from attentions
            contacts = out["contacts"]
            contacts = torch.sigmoid(contacts).unsqueeze(1)  # (B, 1, L, L)

        return rep, contacts
import torch
from src.models.components.esm import pretrained


class ESM2Backbone(torch.nn.Module):
    """
    ESM2 protein language model backbone with optional contact prediction.
    
    Args:
        model_name: ESM2 model variant (e.g., "esm2_t33_650M_UR50D")
        finetune: Whether to fine-tune ESM2 parameters
        return_contacts: Whether to return ESM2's built-in contact predictions
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
        
        self.embedding_cache = {} if not finetune else None
        
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, seq_list, device):
        if self.embedding_cache is None:
            return self._forward_no_cache(seq_list, device)
        else:
            return self._forward_with_cache(seq_list, device)

    def _forward_with_cache(self, seq_list, device):
        """Cache embeddings when model is frozen (no fine-tuning)"""
        batch_embeddings = []
        batch_contacts = []
        to_compute = []
        to_compute_indices = []
        
        # Check cache and collect what needs computing
        for i, (name, seq) in enumerate(seq_list):
            key = f"{name}_{seq}"
            contacts_key = f"{key}_contacts"
            
            if key in self.embedding_cache:
                batch_embeddings.append(self.embedding_cache[key].to(device))
                batch_contacts.append(self.embedding_cache[contacts_key].to(device))
            else:
                batch_embeddings.append(None)  # Placeholder
                to_compute.append((name, seq))
                to_compute_indices.append(i)
                batch_contacts.append(None)
        
        # Compute uncached embeddings in a batch
        if to_compute:
            with torch.no_grad():
                _, _, tokens = self.batch_converter(to_compute)
                tokens = tokens.to(device)
                out = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=True)
                rep = out["representations"][self.model.num_layers][:, 1:-1, :]
                contacts = out["contacts"][:, 1:-1, 1:-1]
                contacts = torch.sigmoid(contacts)  # Convert logits to probabilities [0, 1]
                contacts = contacts.unsqueeze(1)
            
            # Cache and insert into batch lists
            for i, idx in enumerate(to_compute_indices):
                name, seq = to_compute[i]
                key = f"{name}_{seq}"
                
                # Cache embeddings
                self.embedding_cache[key] = rep[i].cpu()
                batch_embeddings[idx] = rep[i]  # Already on device

                contacts_key = f"{key}_contacts"
                self.embedding_cache[contacts_key] = contacts[i].cpu()
                batch_contacts[idx] = contacts[i]
        
        # Pad embeddings to same length
        max_len = max(emb.shape[0] for emb in batch_embeddings)
        d_model = batch_embeddings[0].shape[1]
        
        padded_emb = []
        for emb in batch_embeddings:
            L = emb.shape[0]
            if L < max_len:
                pad = torch.zeros((max_len - L, d_model), device=device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            padded_emb.append(emb)
        
        rep_out = torch.stack(padded_emb)  # (B, max_len, D)
        
        # Pad contacts if needed
        padded_contacts = []
        for cont in batch_contacts:
            _, L1, L2 = cont.shape  # cont is (1, L, L)
            if L1 < max_len or L2 < max_len:
                # Pad to (1, max_len, max_len)
                pad_cont = torch.zeros((1, max_len, max_len), device=device, dtype=cont.dtype)
                pad_cont[:, :L1, :L2] = cont
                cont = pad_cont
            padded_contacts.append(cont)  # Keep as (1, L, L)
        
        # Stack along new batch dimension: list of (1, L, L) → (B, 1, L, L)
        contacts_out = torch.stack(padded_contacts, dim=0)  # (B, 1, max_len, max_len)
        return rep_out, contacts_out
    
    def _forward_no_cache(self, seq_list, device):
        """Standard forward pass (used when fine-tuning)"""
        _, _, tokens = self.batch_converter(seq_list)
        tokens = tokens.to(device)
        
        # Get representations and optionally contacts
        out = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=True)
        rep = out["representations"][self.model.num_layers][:, 1:-1, :]  # Remove BOS/EOS

        # ESM2 returns contact logits from attention maps
        # Shape: (B, L+2, L+2) with BOS/EOS tokens
        contacts = out["contacts"][:, 1:-1, 1:-1]  # Remove BOS/EOS
        contacts = torch.sigmoid(contacts)  # Convert logits to probabilities [0, 1]
        contacts = contacts.unsqueeze(1)  # Add channel dim: (B, 1, L, L)
        return rep, contacts
    
    def clear_cache(self):
        """Clear the embedding cache (useful for memory management)"""
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
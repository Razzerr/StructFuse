import torch


class PairFeatures(torch.nn.Module):
    def __init__(self, d_model, d_pair=128):
        super().__init__()
        self.proj_i = torch.nn.Linear(d_model, d_pair, bias=True)
        self.proj_j = torch.nn.Linear(d_model, d_pair, bias=True)
        
        # Bilinear weight: creates d_pair separate bilinear maps
        self.bilinear_weight = torch.nn.Parameter(torch.randn(d_pair, d_pair, d_pair) * 0.02)
        
        # Instance normalization to reduce row/column artifacts
        self.norm = torch.nn.InstanceNorm2d(d_pair, affine=True)
        
        # Add 3x3 conv to mix spatial features and break checker patterns
        # This helps mix row/column dependencies before the final projection
        self.mix = torch.nn.Conv2d(d_pair, d_pair, 3, padding=1, bias=False)
        
        # Final 1x1 projection
        self.out = torch.nn.Conv2d(d_pair, d_pair, 1, bias=True)
    
    def forward(self, h):
        ui = self.proj_i(h)  # (B, L, d_pair)
        vj = self.proj_j(h)  # (B, L, d_pair)
        
        # Bilinear interaction: for each output channel k, compute ui @ W_k @ vj^T
        feat = torch.einsum('bli,ijk,bmj->bklm', ui, self.bilinear_weight, vj) # (B, d_pair, L, L)
        
        # Normalize to break row/column correlations
        feat = self.norm(feat)
        feat = torch.relu(feat)
        
        # Mix spatially to further reduce checker patterns
        feat = self.mix(feat)
        feat = torch.relu(feat)
        
        return self.out(feat)  # (B, d_pair, L, L)

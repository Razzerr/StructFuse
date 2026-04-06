import torch


class PairFeatures(torch.nn.Module):
    def __init__(self, d_model, d_pair=128, rank=32):
        super().__init__()
        self.proj_i = torch.nn.Linear(d_model, d_pair, bias=True)
        self.proj_j = torch.nn.Linear(d_model, d_pair, bias=True)
        
        # Low-rank bilinear: replaces cubic W[d,d,d] (~2M params) with
        # two rank-r projections + output projection (~16K params at rank=32).
        # Memory: O(r * L²) instead of O(d² * L²) intermediate tensors.
        self.bilinear_left = torch.nn.Linear(d_pair, rank, bias=False)
        self.bilinear_right = torch.nn.Linear(d_pair, rank, bias=False)
        self.bilinear_out = torch.nn.Conv2d(rank, d_pair, 1, bias=True)
        
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
        
        # Low-rank bilinear interaction:
        # Project to rank-r, outer product, then project to d_pair.
        # O(r * L²) memory instead of O(d² * L²) for the cubic einsum.
        left = self.bilinear_left(ui)    # (B, L, r)
        right = self.bilinear_right(vj)  # (B, L, r)
        # Outer product via broadcasting — avoids einsum that Inductor struggles with
        feat = left[:, :, None, :] * right[:, None, :, :]  # (B, L, L, r)
        feat = feat.permute(0, 3, 1, 2)                     # (B, r, L, L)
        feat = self.bilinear_out(feat)   # (B, d_pair, L, L)
        
        # Normalize to break row/column correlations
        feat = self.norm(feat)
        feat = torch.relu(feat)
        
        # Mix spatially to further reduce checker patterns
        feat = self.mix(feat)
        feat = torch.relu(feat)
        
        return self.out(feat)  # (B, d_pair, L, L)

<div align="center">

# StructFuse: Retrieval-Based Structural Fusion for Protein Contact Prediction

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

**Author:** Michał Budnik  
**Date:** October 2025

</div>

---

## 🎯 Overview

This project extends **ESM2's contact prediction capabilities** by incorporating structural information from homologous proteins through template-based fusion. The approach combines ESM2's learned sequence representations with template-derived structural priors using a multi-scale neural architecture.

### What is Contact Prediction?

Protein contact prediction aims to identify pairs of amino acid residues that are spatially close in a protein's 3D structure (< 8 Angstroms), given only the sequence. Accurate contact predictions are crucial for:
- Protein structure prediction
- Understanding protein folding
- Drug design and protein engineering

### Why This Approach?

Pure sequence-based models like ESM2 learn evolutionary patterns but may miss structural constraints. By retrieving and fusing information from structurally similar templates, we leverage both:
- **Evolutionary information** (from ESM2 embeddings)
- **Structural priors** (from homologous protein structures)

This hybrid approach achieves **3-4× improvement** over ESM2-only baselines.

---

## 🏆 Key Results

| Model | P@L ↑ | P@L/2 ↑ | P@L/5 ↑ | Precision ↑ | Recall ↑ | F1 ↑ | Training Time till best model |
|-------|-------|---------|---------|-----------|--------|------|---------------|
| ESM2-only (8M) | 0.183 | 0.201 | 0.208 | 0.235 | 0.044 | 0.074 | N/A |
| ESM2-only (650M) | 0.249 | 0.275 | 0.294 | 0.224 | 0.172 | 0.195 | N/A |
| base | 0.765 | 0.804 | 0.837 | 0.686 | 0.649 | 0.667 | 2h 30m |
| axial | 0.781 | 0.815 | 0.844 | 0.710 | 0.677 | 0.694 | 4h 45m |
| trufor | 0.737 | 0.781 | 0.819 | 0.680 | 0.622 | 0.649 | 1h 55m* |
| **trufor_axial** | **0.789** | **0.822** | **0.851** | **0.734** | **0.683** | **0.708** | 5h 10m |

**Best Model:** `trufor_axial` achieves **F1=0.708** and **P@L=0.789** on TS115 test set.

*Note: trufor training collapsed at epoch 15; best checkpoint saved early.

### Key Findings

✅ **Template fusion provides 3-4× improvement** over ESM2-only baselines  
✅ **Axial attention consistently outperforms CNNs** across all fusion types  
✅ **TruFor cross-attention achieves best results** when stable (trufor_axial)  
⚠️ **Training instability** observed in trufor variant (collapsed after epoch 15)

---

## 🔬 Method

The pipeline consists of five main stages:

### 1. Template Retrieval
- **FAISS-based similarity search** over ESM2 embeddings
- Retrieves **k=4** most similar homologous structures
- **Data leakage prevention:** Test proteins excluded from template database
- **Same-PDB filtering:** Templates from same structure (different chains) are filtered

### 2. Template Processing
- **Needleman-Wunsch alignment** using Parasail library for fast computation
- **BLOSUM62 scoring matrices** for amino acid similarity weighting
- Contact scores adjusted based on alignment quality

### 3. Feature Extraction
- **ESM2 token embeddings** (frozen, pretrained)
- **ESM2 attention-based contacts** from attention maps
- **Template contact features** from aligned homologs

### 4. Feature Fusion
Two fusion strategies implemented:

**Standard Fusion:**
- Direct concatenation with learned projection
- Gated residual connections
- Simpler, more stable training

**TruFor Fusion:**
- Transformer-based cross-attention
- Cross-modal attention between ESM2 and template features
- Higher capacity but requires careful tuning
- Inspired by TruFor image forensics [3]

### 5. Contact Prediction
Two decoder architectures:

**Dilated CNN:**
- Deep convolutional layers with increasing dilation rates
- Efficient for local patterns
- Faster training

**Axial Attention:**
- Factorized row/column attention
- Captures long-range interactions effectively
- AlphaFold-inspired [2]
- Better performance but slower training

---

## 💾 Installation

#### Quick Setup (Recommended)

```bash
# Clone repository
git clone <your-repo-url>
cd https://github.com/Razzerr/StructFuse

# Create conda environment with all dependencies
conda env create -f environment.yaml

# Activate environment
conda activate structfuse
```

The `environment.yaml` automatically:
- Installs Python 3.10
- Installs PyTorch 2.8 with CUDA support via conda
- Installs all other dependencies from `requirements.txt` via pip

#### Manual Setup (Alternative)

```bash
# Create conda environment
conda create -n structfuse python=3.10
conda activate structfuse

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt
```


---

## 📊 Data Preparation

### Step 1: Organize PDB Files

```bash
# Place all PDB files (train + test) in a single directory
mkdir -p data/pdb
# Copy your .pdb files to data/pdb/
```

### Step 2: Build Contact Maps

```bash
# Extract contact maps from PDB structures
# This parses Cα coordinates and computes pairwise distances
python scripts/build_contacts.py

# Output: data/processed/*.npz (cached contact maps)
```

**What it does:**
- Parses PDB files to extract Cα coordinates
- Computes pairwise distances between residues
- Creates binary contact maps (threshold: 8 Angstroms)
- Caches results as `.npz` files for fast loading

### Step 3: Build FAISS Index

```bash
# Build FAISS index for template retrieval
# Default: ESM2-8M (faster, lower memory)
python scripts/build_index.py

# For ESM2-650M (higher quality, more resources):
python scripts/build_index.py model.esm_model_name=esm2_t33_650M_UR50D
```

**What it does:**
- Generates ESM2 embeddings for all proteins
- Builds FAISS index for fast similarity search
- Respects train/val/test splits (test proteins excluded)
- Caches embeddings and index to disk

**Output:** `data/index_t6/` (or `data/index_t33/` for 650M)
- `embeddings.npy` - ESM2 embeddings
- `faiss.index` - FAISS index file
- `ids.json` - Protein ID mapping

### Step 4: Verify Data Integrity

```bash
# Run data leakage checks
python scripts/check_data_leakage.py
```

**Checks performed:**
- No overlap between train/val/test splits
- Test proteins not in FAISS index
- Same-PDB filtering works correctly
- All split files are consistent

✅ **All checks should pass** before training!

---

## 🚀 Training

### Quick Start (Best Model)

```bash
# Train the best performing model (TruFor + Axial Attention)
python src/train.py experiment=trufor_axial
```

### Available Experiments

```bash
# Standard fusion + Dilated CNN (baseline)
python src/train.py experiment=base

# Standard fusion + Axial Attention
python src/train.py experiment=axial

# TruFor fusion + Dilated CNN
python src/train.py experiment=trufor

# TruFor fusion + Axial Attention (best performance)
python src/train.py experiment=trufor_axial

# ESM2-only baseline (no templates)
python scripts/eval_esm2_baseline.py experiment=esm2_only

# For ESM2-650M baseline:
python scripts/eval_esm2_baseline.py experiment=esm2_only model=esm2_t33_650M
```

### Training Configuration

All experiments use consistent hyperparameters:
- **Optimizer:** AdamW (lr=2e-4, weight_decay=1e-2)
- **Batch size:** 8 (axial), 16 (others)
- **Max epochs:** 50 with early stopping (patience=5)
- **Loss:** Balanced BCE + Label Smoothing + Tversky
- **Augmentation:** Random 256×256 crops
- **Hardware:** Single GPU (tested on RTX 2070)

---

## 📈 Evaluation

### Evaluate Trained Model

```bash
# Evaluate on test set
python src/eval.py experiment=<experiment_name> ckpt_path=<path_to_checkpoint>

# Example:
python src/eval.py experiment=<experiment_name> ckpt_path=logs/train/runs/2025-10-19_12-30-45/checkpoints/best.ckpt
```

**Metrics computed:**
- **P@L, P@L/2, P@L/5:** Precision at top-L predictions (L = sequence length)
- **Precision, Recall, F1:** Standard classification metrics
- **Optimal threshold:** F1-maximizing threshold from validation set

**Outputs:**
- Metrics printed to console and saved to logs
- Contact map visualizations for each test protein
- Precision-recall curves

### Visualization

Contact maps are automatically generated during test and saved to:
```
logs/<experiment>/runs/YYYY-MM-DD_HH-MM-SS/test_visualizations/
```

Each visualization includes:
- Ground truth contact map
- Predicted contact probabilities
- Binary predictions (with optimal threshold)
- Side-by-side comparison

---

## 🔧 Technical Details

### Hardware Requirements

**Tested:**
- GPU: NVIDIA RTX 2070 (8GB VRAM)
- RAM: 16GB (or 128 for full cache utilization)
- Storage: 10GB for data + models

### Optimization & Caching

To enable efficient training on consumer hardware, several optimizations are implemented:

**Preprocessing & Caching:**
- Contact maps extracted from PDB and cached as `.npz` files
- ESM2 embeddings precomputed and stored
- Needleman-Wunsch alignments cached for template-target pairs

**Impact:**
- Batch loading time: ~10s → <1s per batch
- Enables training on 8GB VRAM GPU
- First epoch slower (cache building), subsequent epochs fast

**Memory Management:**
- Random crops (256×256) instead of full protein sequences
- Gradient checkpointing available (set `model.use_checkpointing=true`)
- Mixed precision training (AMP) enabled by default

### Loss Function Design

**Balanced Binary Cross-Entropy (BCE):**
```python
pos_weight = (n_neg / n_pos) * pos_weight_scale  # scale = 0.8
```

**Label Smoothing:**
- Prevents overconfident predictions
- Smoothing factor: 0.05

**Symmetry Enforcement:**
- Contact maps are symmetric (Cij = Cji)
- Logits symmetrized before computing loss: `logits = (logits + logits.T) / 2`

**Tversky Loss (Auxiliary):**
- Generalization of Dice loss
- Asymmetric weighting: α=0.7 (FP), β=0.3 (FN)
- Focal modulation: γ=1.0 for hard examples
- Combined as: `Total Loss = BCE + 0.3 × Tversky`

**Threshold Optimization:**
- After each validation epoch, optimize threshold by maximizing F1
- Optimal threshold saved with model checkpoint
- Used during testing for binary predictions

### Data Leakage Prevention

Multiple safeguards ensure fair evaluation:

1. **Pre-split indexing:** FAISS index built only on train+val splits
2. **Same-PDB filtering:** Templates from same PDB (different chains) filtered
3. **Validation script:** `check_data_leakage.py` verifies zero overlap
4. **Deterministic splits:** Fixed random seed (42) for reproducibility

**Verification:**
```bash
python scripts/check_data_leakage.py
# All checks must pass ✓
```

### Architecture Variants

**Fusion Strategies:**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| Standard | Concatenation + projection | Stable, fast | Limited cross-modal interaction |
| TruFor | Cross-attention fusion | Higher capacity, best performance | Training instability |

**Decoder Heads:**

| Head | Description | Receptive Field | Speed |
|------|-------------|-----------------|-------|
| DilatedCNN | Multi-scale dilated convolutions | Local to medium | Fast |
| AxialAttention | Factorized row/column attention | Global | Slower |

**Model Combinations:**
- `base` = Standard + DilatedCNN
- `axial` = Standard + AxialAttention
- `trufor` = TruFor + DilatedCNN
- `trufor_axial` = TruFor + AxialAttention (best)

---

## 🔁 Reproducibility

### Deterministic Training

All experiments use fixed random seeds for reproducibility:

```python
seed: 42
deterministic: true  # PyTorch deterministic mode
```


## 📖 References

- **[1] ESM2:** Lin, Z., et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*, 379(6637), 1123-1130.

- **[2] AlphaFold2:** Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596(7873), 583-589.

- **[3] TruFor:** Guillaro, F., et al. (2023). "TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization." CVPR 2023.

- **[4] FAISS:** Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 7(3), 535-547.

- **[5] Parasail:** Daily, J. (2016). "Parasail: SIMD C library for global, semi-global, and local pairwise sequence alignments." *BMC Bioinformatics*, 17(1), 81.

- **[6] BLOSUM:** Henikoff, S., & Henikoff, J. G. (1992). "Amino acid substitution matrices from protein blocks." *PNAS*, 89(22), 10915-10919.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{budnik2025,
  author = {Budnik, Michał},
  title = {StructFuse: Retrieval-Based Structural Fusion for Protein Contact Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Razzerr/StructFuse}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Lightning** for the training framework
- **Hydra** for elegant configuration management
- **Meta AI** for ESM2 models
- **Facebook Research** for FAISS
- **ashleve** for the Lightning-Hydra template

---

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

Final paper results are intentionally not reported here until the complete
experiment matrix has been rerun under the frozen evaluation protocol.

---

## 🏆 Key Results


### Key Findings

---

## 🔬 Method

The pipeline consists of five main stages:

### 1. Template Retrieval
- **FAISS-based similarity search** over ESM2 embeddings
- Retrieves **k=4** most similar homologous structures
- The structural index contains all available structures for inference
- **Leakage prevention:** same-protein and same-cluster templates are always
  filtered; validation/test holdouts are additionally filtered from retrieval
  during training

### 2. Template Processing
- **Needleman-Wunsch alignment** using Parasail library for fast computation
- **BLOSUM62 scoring matrices** for amino acid similarity weighting
- Contact scores adjusted based on alignment quality

### 3. Feature Extraction
- **ESM2 token embeddings** (frozen, pretrained)
- **ESM2 attention-based contacts** from attention maps
- **Template contact features** from aligned homologs

### 4. Feature Fusion

The final model uses grouped feature fusion for ESM2 representations,
attention-derived contacts, template contact priors, coverage counts, and
template distance bins. Standard concatenation and TruFor-style cross-attention
remain implemented as controlled ablations.

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
git clone https://github.com/Razzerr/StructFuse
cd StructFuse

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

---

## 🚀 Training

### Quick Start (Best Model)


### Optimization & Caching

### Loss Function Design


### Data Leakage Prevention


### Architecture Variants

**Fusion Strategies:**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| Grouped | Feature-group projections followed by learned fusion | Final model; preserves source structure | More components |
| Standard | Concatenation + projection | Simple and fast | Controlled ablation |
| TruFor | Cross-attention fusion | Higher capacity | Controlled ablation; more expensive |

**Decoder Heads:**

| Head | Description | Receptive Field | Speed |
|------|-------------|-----------------|-------|
| DilatedCNN | Multi-scale dilated convolutions | Local to medium | Fast |
| AxialAttention | Factorized row/column attention | Global | Slower |

The final configuration is grouped fusion with the `axial_tri` pair head
(axial attention plus triangle multiplicative updates). Dilated, standard
fusion, TruFor fusion, and no-triangle variants are retained as ablations.

---

## 🔁 Reproducibility

### Deterministic Training


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

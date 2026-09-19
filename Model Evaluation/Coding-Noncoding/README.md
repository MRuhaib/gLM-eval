# Coding-Noncoding Classification & Feature Attribution

This subfolder contains our most recent and critical line of research: **investigating why and how frozen Genome Language Model (gLM) embeddings separate protein-coding and non-coding DNA**, and performing **token-level feature attribution and representational alignment (CKA)** to interpret the internal mechanisms driving this separation.

---

## Research Overview & Motivation

One of the most remarkable emergent properties of pretrained gLMs is their ability to separate protein-coding sequences from non-coding/intergenic sequences in embedding space without task-specific fine-tuning. 

When evaluated with simple linear or random forest probes:
- In an **80-20 train-test split (one-shot setting)**, multiple models achieve $>90\%$ classification accuracy.
- Even in a **1-99 train-test split (pseudo zero-shot setting)**, where the classifier sees only $1\%$ of the data, models maintain robust discriminability.

This raises crucial interpretability questions:
1. *What biological sequence motifs or token patterns are the models using to make this distinction?*
2. *Is the model recognizing open reading frames, codon structure, GC content, or specific regulatory k-mers?*
3. *Do distinct gLM architectures (Transformers, Hyena convolutions, BigBird) converge on the same representational geometry?*

To answer these questions, this module provides classification probes, end-to-end gradient attribution via **Captum**, Layer-wise Relevance Propagation via **LXT**, and cross-model **Centered Kernel Alignment (CKA)**.

---

## Directory Organization

```
Coding-Noncoding/
├── classifier.ipynb               # Random Forest classification probes & clustering ARI verification
├── Attribution/
│   ├── Captum/
│   │   ├── glm-classification-attribution.ipynb # End-to-end LayerIntegratedGradients pipeline
│   │   └── attribution_analysis.ipynb           # Comprehensive attribution analytics (GC, entropy, k-mers)
│   └── LXT/
│       └── attribution_analysis copy.ipynb      # Layer-wise Relevance Propagation (LRP) for transformers
└── CKA/
    ├── cka_gLM_alignment.ipynb    # Batched linear Centered Kernel Alignment across models
    └── results/
        └── cka_analysis.ipynb     # Pairwise CKA matrices & cross-architecture representational similarity
```

---

## Methodological Breakdown

### 1. Classification Probes & Clustering
`classifier.ipynb`
- **Datasets Evaluated**:
  - Human reference genome GRCh38 (20,000 coding CDS vs. 20,000 intergenic sequences).
  - Genomic Benchmarks (`coding_vs_intergenomic_seqs`, 50,000 fixed-length 200 bp sequences each).
- **Probing Regimes**:
  - **One-Shot / Supervised**: 80-20 and 75-25 train-test splits.
  - **Pseudo Zero-Shot**: 1-99 train-test split (training on 1% and evaluating on the remaining 99%).
- **Unsupervised Verification**: Evaluates K-Means, DBSCAN, and Birch clustering on the raw embeddings; computes Adjusted Rand Index (ARI) against true coding/non-coding labels.
- **Dimensionality Reduction**: Generates UMAP projections showing explicit cluster separation (e.g., highly pronounced in GENA-LM and GPN).

---

### 2. Feature Attribution via Captum (`LayerIntegratedGradients`)
Located in `Attribution/Captum/`:

`glm-classification-attribution.ipynb`
1. **End-to-End Architecture**:
   - Trains a multi-layer perceptron probe (`CodingNonCodingClassifier`: input dim $\to$ 256 $\to$ 64 $\to$ 2).
   - Packages the frozen foundation gLM and probe into a unified `GenomeLMWithClassifier(nn.Module)` that accepts `inputs_embeds` to support gradient-based backpropagation.
2. **Attribution Engine**:
   - Applies Captum's `LayerIntegratedGradients` targeting the token embedding layer of the gLM.
   - Computes attributions relative to a zero/baseline embedding across 50 integration steps.
   - Segregates attributions for correct vs. incorrect predictions to isolate failure modes.

`attribution_analysis.ipynb`
Performs deep-dive interpretability on the resulting token attributions:
- **High-Attribution K-mers**: Identifies top positive tokens (driving coding decisions) and top negative tokens (driving non-coding decisions).
- **GC Content vs. Attribution**: Regresses attribution magnitude against sequence GC percentage to test for composition bias.
- **Attribution vs. Token Frequency**: Verifies whether high attributions are driven by rare specialized tokens or frequent baseline tokens.
- **Attribution Entropy**: Measures whether decision confidence is distributed uniformly across the sequence or concentrated in sparse localized motifs.
- **Positional & Repeating Pattern Analysis**: Tests for periodic signals (e.g., 3-base periodicity in coding ORFs) and repetitive elements (e.g., dinucleotide tandem repeats like `TGTGTG`).

---

### 3. Layer-wise Relevance Propagation (LXT)
Located in `Attribution/LXT/`:
- Uses the Local eXplanations for Transformers (LXT) framework to propagate conservation and relevance backward through multi-head attention and feed-forward layers.
- Serves as an independent cross-validation for gradient-based Captum attributions.

---

### 4. Centered Kernel Alignment (CKA)
Located in `CKA/`:

`cka_gLM_alignment.ipynb` & `cka_analysis.ipynb`
- Computes **Linear Centered Kernel Alignment (CKA)** between feature representations of different models:
  $$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$
  where $K = X X^T$ and $L = Y Y^T$ are centered Gram matrices.
- **Key Question**: Do structurally distinct models (e.g., HyenaDNA's long convolutions vs. DNABERT's bidirectional self-attention vs. BigBird's sparse attention) learn geometrically equivalent sub-manifolds for coding sequences?

---

## How to Run

### 1. Classification & Probing
Run the Jupyter notebook:
```bash
jupyter notebook "Coding-Noncoding/classifier.ipynb"
```

### 2. Captum Token Attribution Pipeline
Open and execute:
```bash
jupyter notebook "Coding-Noncoding/Attribution/Captum/glm-classification-attribution.ipynb"
```
Followed by the downstream analysis:
```bash
jupyter notebook "Coding-Noncoding/Attribution/Captum/attribution_analysis.ipynb"
```

### 3. CKA Representational Alignment
Compute and plot CKA matrices:
```bash
jupyter notebook "Coding-Noncoding/CKA/cka_gLM_alignment.ipynb"
jupyter notebook "Coding-Noncoding/CKA/results/cka_analysis.ipynb"
```

---

## Summary of Findings

1. **Robust Linear Separability**: The coding vs. non-coding boundary is intrinsically encoded in the geometry of frozen embeddings, requiring minimal supervision to achieve $>90\%$ probing accuracy.
2. **Attribution Drivers**: Captum attributions show that models focus heavily on start/stop codon motifs and reading-frame triplet structures in coding regions, while non-coding classification is often influenced by repetitive low-complexity sequences and structural GC skews.
3. **Cross-Architecture Alignment**: CKA scores between transformer-based models (DNABERT, NT, GENA-LM) and convolutional models (HyenaDNA) are exceptionally high in the coding subspace, showing that foundational models converge on invariant genomic representations regardless of model architecture.

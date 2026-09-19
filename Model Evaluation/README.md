# Intrinsic Evaluation of DNA Embeddings in Genome Language Models (gLMs)

## Overview

Large language model (LLM) architectures applied to biological sequences have yielded a diverse suite of **Genome Language Models (gLMs)** such as DNABERT-2, Nucleotide Transformer, GENA-LM, GPN, GROVER, and HyenaDNA. While these foundation models achieve impressive performance on supervised downstream tasks (promoter detection, splice-site identification, variant effect prediction), their evaluation has historically remained **extrinsic**; measuring task-specific benchmark metrics rather than characterizing what the representations themselves inherently encode.

Recent literature highlights that pretraining performance does not always correlate with downstream biological fidelity, and gLMs can behave like black boxes prone to representational biases.

This repository provides an **end-to-end, task-independent intrinsic evaluation suite** for gLM representations. By characterizing frozen, un-fine-tuned embedding spaces in isolation, we systematically probe how models represent sequence similarity, evolutionary taxonomy, mutational perturbations, codon degeneracy, and functional genomic partitions.

---

## Core Hypotheses & Theoretical Framework

The evaluation framework investigates three primary intrinsic hypotheses and a mechanistic feature attribution study:


### 1. Sequence Similarity ($\mathcal{H}_1$)
*Similar sequences should yield geometrically proximal embeddings.*  
For homologous genes across 1,011 *Saccharomyces cerevisiae* strains, the pairwise Needleman-Wunsch global alignment score $s_{ij}$ is compared against the Euclidean distance $d_{ij}$ and cosine distance of their frozen gLM embeddings:
$$d_{ij} \propto \frac{1}{s_{ij}} \quad \iff \quad \rho(d_{ij}, s_{ij}) < 0$$
Evaluated across both 7 canonical yeast genes (600 to 4,000 bp) and a uniformly sampled benchmark of 100 genes (50 to 5,000 bp).

### 2. Embedding Contextuality ($\mathcal{H}_2$)
*Homologous sequences in different biological/evolutionary contexts should occupy distinct clusters.*  
Using a single conserved gene (e.g., *YAL001C*), we test whether models can infer the **phylogenetic clade** of a strain (out of 30 distinct clades cataloged by Peter et al., 2018). Clustering algorithms (K-Means, Agglomerative, Birch, DBSCAN) are evaluated using Adjusted Rand Index (ARI) and Silhouette Scores.  
**Control Experiment**: Genotype matrices (VCFs) and SNP one-hot encodings are compared against gLM embeddings using Procrustes analysis, nearest-neighbor graph overlap, and clustering metrics to determine whether gLMs extract deeper evolutionary signals than raw single-nucleotide variants alone.

### 3. Embedding Robustness & Codon Degeneracy ($\mathcal{H}_3$)
*The magnitude of embedding displacement caused by nucleotide substitution should reflect functional impact.*  
Synthetic point mutations are systematically introduced at increasing rates ($5\%$ to $50\%$). Across models, **non-coding sequences exhibit systematically larger embedding shifts than coding sequences** ($p < 10^{-15}$, Kolmogorov-Smirnov test).  
This asymmetry is explained by **codon degeneracy**: redundant codons buffer coding sequences against synonymous mutations, particularly at the **wobble (3rd) codon position**. We validate this using exhaustive single-codon and $n$-codon point mutations on Genomic Benchmarks.

### 4. Coding vs. Non-coding Separability & Mechanistic Attribution
Frozen gLM embeddings cleanly separate protein-coding from non-coding DNA (>90% accuracy with simple linear/random forest probes in one-shot and pseudo zero-shot splits). To understand the mechanistic basis:
- **Captum (`LayerIntegratedGradients`)**: Attributions quantify token-level importance, isolating top k-mers, GC content bias, and positional significance.
- **Layer-wise Relevance Propagation (LXT)**: Local explanation of transformer layers.
- **Centered Kernel Alignment (CKA)**: Quantifies representational alignment across distinct model architectures.

---

## Evaluated Genome Language Models

The suite evaluates 10+ gLMs spanning diverse architectures, context windows, and tokenization paradigms:

| Model Name | Architecture | Parameter Count | Tokenization Strategy | Context Window | Pretraining Corpus |
|:---|:---:|:---:|:---:|:---:|:---|
| **DNABERT-2** | BERT (Transformer Encoder) | 117M | Byte-Pair Encoding (BPE) | 512 bp | Human + Multispecies |
| **DNABERT-S** | BERT (Transformer Encoder) | 117M | Byte-Pair Encoding (BPE) | 512 bp | Human + Multispecies |
| **Nucleotide Transformer (NT-500M)** | BERT (Transformer Encoder) | 500M | 6-mer overlapping tokens | 1,000 bp | Human Reference |
| **Nucleotide Transformer (NT-2.5B-1000G)** | BERT (Transformer Encoder) | 2.5B | 6-mer overlapping tokens | 1,000 bp | 1,000 Genomes (Human) |
| **Nucleotide Transformer (NT-2.5B-MS)** | BERT (Transformer Encoder) | 2.5B | 6-mer overlapping tokens | 1,000 bp | 850+ Species (Multispecies) |
| **GENA-LM** | BigBird (Sparse Attention) | 336M | Byte-Pair Encoding (BPE) | 36,000 bp | Human T2T |
| **GPN (Brassicales)** | CNN-Transformer Hybrid | 65M | Character-level | 512 bp | Brassicales / Whole Genomes |
| **GROVER** | BERT (Transformer Encoder) | ~110M | Byte-Pair Encoding (BPE) | 510 bp | Human Reference (GRCh38) |
| **HyenaDNA-tiny (1k, 16k)** | Hyena Operator (Convolutional) | 0.4M - 1.5M | Character-level | 1k - 16k bp | Human Reference |
| **HyenaDNA-small (32k)** | Hyena Operator (Convolutional) | 2.5M | Character-level | 32k bp | Human Reference |
| **HyenaDNA-medium (160k, 450k)** | Hyena Operator (Convolutional) | 4.5M - 6.5M | Character-level | 160k - 450k bp | Human Reference |
| **HyenaDNA-large (1M)** | Hyena Operator (Convolutional) | 6.5M | Character-level | 1,000,000 bp | Human Reference |

---

## Repository Structure & Submodule Index

The codebase is modularly organized into 8 functional subdirectories, each dedicated to a distinct phase of the intrinsic evaluation pipeline:

```
Model Evaluation/
├── Clade Clustering/          # Phylogenetic clade clustering, UMAP visualization & SNP control experiments
├── Coding-Noncoding/          # Classification probes, Captum LayerIntegratedGradients & CKA alignment
├── Downstream Evaluation/     # Quantitative phenotype prediction across 1,011 yeast strains
├── Embeddings Evaluation/     # Sequence alignment vs. embedding distance correlation & ranking metrics
├── Model Inference/           # Unified feature extraction engine for single genes and whole genomes
├── Mutation Sensitivity/      # Synthetic point mutation sensitivity, codon degeneracy & wobble analysis
├── Sequence Alignment/        # Pairwise Needleman-Wunsch (Biopython/Biotite) & Hamming distance aligners
├── Yeast Gene Selection/      # Curation, length sorting & uniform sampling of genes across 1,011 strains
├── report.tex                 # LaTeX source code for the comprehensive academic report
├── UGRC_2026_Report_...pdf    # Compiled final research project report
└── .gitignore                 # Exclusion rules safeguarding large matrices and raw embeddings
```



## Primary Datasets

1. **1,011 *Saccharomyces cerevisiae* Strains (Peter et al., 2018)**:
   - Complete genome assemblies and CDS gene sequences for 1,011 isolates of baker's yeast spanning 30 ecologically and geographically diverse clades.
   - Ground-truth phenotypic growth matrix across environmental stress factors (`pheno_matrix.txt`).
2. ***Sarcopterygii matsunami* Conserved Non-Coding Elements (Inoue & Saitou, 2021)**:
   - Ultra-conserved non-coding elements from dbCNS, providing clean non-coding baseline sequences matching model context lengths.
3. **Human Reference Genome (GRCh38 / Ensembl)**:
   - 20,000 coding CDS sequences and 20,000 non-coding intergenic sequences sampled uniformly to test cross-species transferability.
4. **Genomic Benchmarks (`coding_vs_intergenomic_seqs`)**:
   - Standardized human benchmark comprising 50,000 fixed-length (200 bp) coding and 50,000 intergenomic sequences.

---

## Key Experimental Results

1. **Alignment Correlation vs. Context Length**:
   - All models exhibit negative correlations between sequence alignment scores and embedding Euclidean distances.
   - HyenaDNA models excel on long sequence inputs (e.g., *YAL026C*, 4,068 bp), maintaining stable correlation coefficients.
   - Transformer models with small context windows (Nucleotide Transformer, GROVER, GPN) show degradation in correlation magnitude as sequence length surpasses their native window.
2. **Evolutionary Clade Conservation**:
   - Embeddings derived from a single gene sequence (*YAL001C*) spontaneously resolve 25–30 distinct clusters corresponding to true strain clades.
   - Control experiments with raw SNP matrices demonstrate that while raw nucleotide variations explain first-order clustering, gLMs capture non-linear sequence dependencies that preserve hierarchical cluster geometry.
3. **Coding vs. Non-Coding Mutation Asymmetry**:
   - Point mutations induce significantly larger Euclidean shifts in non-coding DNA compared to coding DNA ($p < 10^{-15}$).
   - Exhaustive codon mutation experiments demonstrate that synonymous mutations at the 3rd (wobble) position produce markedly smaller embedding shifts than missense mutations in coding ORFs.
4. **Token Attribution Insights**:
   - Captum `LayerIntegratedGradients` reveals that coding classification is driven by open reading frame periodicity, specific GC-rich regulatory k-mers, and start/stop codon motifs rather than uniform nucleotide frequency.
   - Linear CKA confirms high cross-architecture alignment on the coding vs. non-coding subspace between BERT, Hyena, and BigBird architectures.

---

## Environment & Prerequisites

### Installation
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/MRuhaib/UGRC-2024.git
cd UGRC-2024/"Model Evaluation"
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### Dependencies
Install core dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets captum biopython biotite
pip install scikit-learn umap-learn scipy numpy pandas matplotlib seaborn tqdm
pip install git+https://github.com/songlab-cal/gpn.git
```

---

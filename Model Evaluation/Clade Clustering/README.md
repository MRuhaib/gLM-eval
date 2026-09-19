# Clade Clustering & Variant Analysis

This subfolder evaluates **Hypothesis 2: Embedding Contextuality**—testing whether Genome Language Models (gLMs) can inherently recover strain evolutionary context (phylogenetic clades) from single gene sequences alone, and provides a rigorous control experiment against raw single-nucleotide polymorphism (SNP) variant matrices.

---

## Biological Context & Motivation

The 1,011 isolates of *Saccharomyces cerevisiae* sequenced by [Peter et al. (2018)](https://doi.org/10.1038/s41586-018-0030-5) are phylogenetically partitioned into **30 distinct clades** reflecting geographical origin (e.g., Wine/European, West African, North American oak) and ecological niche. 

While full-genome alignment naturally resolves these clades, whole yeast genomes (~12 Mb) far exceed the context windows of most gLMs (500 bp to 36,000 bp). Here, we test whether un-fine-tuned gLM embeddings derived from **a single conserved gene** (such as *YAL001C*, ~4,500 bp) retain sufficient strain-specific evolutionary signal to spontaneously cluster isolates by their parent clade.

---

## Directory Organization

```
Clade Clustering/
├── Clustering/
│   ├── cluster.py                 # Multi-algorithm clustering & ARI/silhouette evaluation
│   ├── hierarchicalCluster.py     # Hierarchical agglomerative clustering & dendrograms
│   └── Results/                   # Output clustering metric plots & parameter curves
├── Data Labelling/
│   ├── read.py                    # Metadata parser mapping 1,011 strains to 30 clades
│   └── metadata2.csv              # Ground-truth strain isolate metadata (Peter et al., 2018)
├── Vizualisation/
│   ├── vizualise.py               # 2D UMAP & t-SNE embedding projection pipelines
│   └── *.png                      # Generated scatter plots across models
└── Variants Analysis/             # Control Experiment: Raw Variant vs. gLM Representation
    ├── 7 genes/
    │   ├── generate_vcf.py        # Calls variants relative to reference strain into VCF
    │   ├── build_snp_onehot.py    # Builds SNP one-hot and genotype matrices from VCFs
    │   ├── variant_umap.py        # Computes UMAP projections on raw genotype matrices
    │   ├── glm_variant_comparison.py # Procrustes, kNN overlap, ARI & Silhouette benchmark
    │   └── glm_vs_oh_embeddings.ipynb# Interactive comparison notebook
    └── 100 genes/
        ├── kaggle_100_genes_encodings.ipynb       # Scaled encoding pipeline for 100 genes
        └── kaggle_100_genes_modelwise_analysis.ipynb# Model-wise metrics across gene lengths
```

---

## Key Workflows & Scripts

### 1. Unsupervised Clade Clustering
`Clustering/cluster.py`
Evaluates how well reduced gLM embeddings partition into known clades using unsupervised clustering:
- **Supported Algorithms**: K-Means, Agglomerative Clustering, Birch, DBSCAN, OPTICS, Affinity Propagation.
- **Evaluation Metrics**:
  - **Adjusted Rand Index (ARI)**: Quantifies agreement between cluster assignments and ground-truth clades ($[-1, 1]$).
  - **Silhouette Score**: Assesses intra-cluster cohesion versus inter-cluster separation.
- **Parameter Sweeps**: Evaluates cluster count ($k \in [2, 50]$) and nearest neighbor thresholds. The highest ARI scores occur at $k \approx 30$, precisely matching the biological clade count.

`Clustering/hierarchicalCluster.py`
Performs Hierarchical Agglomerative Clustering (HAC) using Ward/Single linkage on pairwise Euclidean distance matrices and plots dendrogram trees.

### 2. Dimensionality Reduction & Visualization
`Vizualisation/vizualise.py`
Projects high-dimensional gLM embeddings into 2D using UMAP (`umap-learn`) and t-SNE (`scikit-learn`):
- Colors points according to the 30 ground-truth clades from `metadata2.csv`.
- Generates 3x3 model grid figures (`allUmaps3x3.png`) showing conserved cluster topologies across architectures.

### 3. The Variant Control Experiment
Located in `Variants Analysis/7 genes/`:

#### Scientific Rationale
If gLM embeddings cluster by clade, does this simply reflect trivial linear Hamming distances between nucleotide substitutions, or does the model capture non-linear, contextual biological structure?
- **Step 1 (`generate_vcf.py`)**: Uses C-backed `difflib.SequenceMatcher` to perform fast pairwise alignment of 1,011 gene sequences against strain 1 (reference), emitting standard VCF files with `CHROM, POS, REF, ALT, GT` fields.
- **Step 2 (`build_snp_onehot.py`)**: Converts VCF genotype calls into:
  - `snp_onehot.npy`: Binary matrix $[N_{\text{strains}}, 4 \times N_{\text{sites}}]$ encoding A, C, G, T states.
  - `vcf_gt_matrix.npy`: Integer matrix $[N_{\text{strains}}, N_{\text{sites}}]$ of ALT allele indices.
- **Step 3 (`variant_umap.py`)**: Constructs UMAP embeddings directly from raw variant matrices using identical hyperparameters.
- **Step 4 (`glm_variant_comparison.py`)**: Compares gLM UMAPs against raw SNP UMAPs using:
  - **Orthogonal Procrustes Distance**: Quantifies global geometric shape disparity.
  - **$k$-Nearest Neighbor ($k$NN) Graph Jaccard Overlap**: Measures local neighborhood conservation.
  - **ARI & Silhouette**: Compares cluster recovery between raw variants and foundation representations.

---

## Execution Instructions

### Running Clade Clustering
```bash
python "Clade Clustering/Clustering/cluster.py"
```

### Running the Variant Control Pipeline
```bash
# 1. Generate VCFs from gene FASTAs
python "Clade Clustering/Variants Analysis/7 genes/generate_vcf.py"

# 2. Build SNP one-hot encodings
python "Clade Clustering/Variants Analysis/7 genes/build_snp_onehot.py"

# 3. Compute UMAP on raw variants
python "Clade Clustering/Variants Analysis/7 genes/variant_umap.py"

# 4. Compare raw variants against gLM embeddings
python "Clade Clustering/Variants Analysis/7 genes/glm_variant_comparison.py"
```

---

## Key Findings

1. **Autonomous Clade Separation**: Unsupervised clustering on frozen embeddings from a single gene recovers 25–30 clusters aligned with true phylogenetic clades.
2. **Topological Invariance**: Cluster shapes in UMAP space are remarkably consistent across distinct architectures (DNABERT-2, GENA-LM, GPN, HyenaDNA), indicating that different pretraining objectives converge on similar phylogenetic signals.
3. **Beyond Raw SNPs**: While raw SNP matrices also reflect clade structure (as clades are defined by shared mutations), gLM representations exhibit superior cluster cohesion and smoother local neighborhood graphs, proving models learn contextual representations rather than memorizing isolated point substitutions.

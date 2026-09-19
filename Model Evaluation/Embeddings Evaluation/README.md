# Embeddings Evaluation: Sequence Similarity Correlation Suite

This subfolder contains the core analytical pipeline for testing **Hypothesis 1: Sequence Similarity**—evaluating whether the geometric proximity of Genome Language Model (gLM) embeddings accurately reflects biological sequence similarity across 1,011 *Saccharomyces cerevisiae* strains.

---

## Theoretical Framework

If a foundation model learns the syntax and grammar of genomic sequences, then **two DNA sequences with high sequence alignment similarity should produce embedding vectors that are close in vector space**.

For any pair of strains $i$ and $j$ possessing homologous sequences:
- Ground-truth sequence similarity is quantified by the **Needleman-Wunsch alignment score** $s_{ij}$ (computed in `Sequence Alignment/`). Higher scores indicate greater sequence identity.
- Embedding dissimilarity is quantified by the **Euclidean distance** $d_{ij} = \|\mathbf{e}_i - \mathbf{e}_j\|_2$ or **Cosine distance** $1 - \cos(\mathbf{e}_i, \mathbf{e}_j)$.

Under our hypothesis:
$$d_{ij} \propto \frac{1}{s_{ij}} \quad \implies \quad \rho(d_{ij}, s_{ij}) < 0$$
where $\rho$ is the Pearson or Spearman correlation coefficient. A stronger negative correlation indicates that the model's latent geometry faithfully preserves single-nucleotide differences and structural alignment.

---

## Directory Organization

```
Embeddings Evaluation/
├── Single Gene Embeddings/        # Correlation analysis on 7 canonical yeast genes
│   ├── main.py                    # Computes Pearson/Spearman correlations & evaluation metrics
│   ├── Embeddings Alignment/
│   │   ├── alignment.py           # Computes 1,011 x 1,011 pairwise Euclidean distance matrices
│   │   ├── distHist.py            # Generates pairwise distance distribution histograms
│   │   ├── fill.py                # Matrix symmetrization utility
│   │   └── Embedding Alignment Scores/
│   │       └── Cosine/
│   │           └── cosineAlignment.py # Computes pairwise Cosine similarity matrices
│   └── Histograms/                # Plots showing distance distribution shifts
├── 100 Genes/                     # Scaled benchmark across 100 genes (50 to 5,000 bp)
│   ├── sequence-alignment.ipynb   # Batch Needleman-Wunsch alignment for 100 genes
│   ├── glm-inference.ipynb        # Embedding generation pipeline for 100 genes
│   └── correlator.ipynb           # Computes correlation vs. gene length regression curves
├── Full Genome Embeddings/        # Full-genome segmented embedding evaluation
│   ├── fullEmbeddingsEval.ipynb   # Windowed full-genome alignment vs. embedding correlation
│   └── embeddingVisualisation.ipynb # 2D visualization of segmented genome representations
└── Ranking Method/                # Relative neighborhood ranking & top-k retrieval
    ├── Sequence Rankings/
    │   └── rankingSequences.py    # Ranks strains by sequence alignment similarity
    └── Embeddings Ranking/
        └── rankingEmbeddings.py   # Ranks strains by embedding proximity and compares ranks
```

---

## Detailed Component Breakdown

### 1. Single Gene Correlation Pipeline (7 Benchmark Genes)
Located in `Single Gene Embeddings/`:
- **Selected Genes**: Evaluates 7 conserved genes of varying lengths:
  - *YAL007C* (648 bp - short)
  - *YAL001C* (4,524 bp - long)
  - *YAL026C* (4,068 bp - long)
  - *YAL038W*, *YAL056C-A*, *YAL060W*, *YBL075C*
- `Embeddings Alignment/alignment.py`: Reads raw $1,011 \times D$ embedding matrices from `Model Inference/Embeddings/`, computes pairwise distances using `scipy.spatial.distance.pdist(embeddings, metric="euclidean")`, and saves the resulting $1,011 \times 1,011$ symmetric distance matrix.
- `Embeddings Alignment/Embedding Alignment Scores/Cosine/cosineAlignment.py`: GPU-accelerated PyTorch cosine similarity matrix builder.
- `main.py`: Loads the pairwise Needleman-Wunsch matrix and the embedding distance matrix, flattens the upper-triangular components, and computes correlation coefficients $\rho$:
  - `evaluateScoreCorr(model, gene)`: Per-strain correlation distributions.
  - `evaluateFlattenedCorr(model, gene)`: Global flattened matrix correlation.

---

### 2. 100-Gene Scaled Benchmark
Located in `100 Genes/`:
- **Objective**: To investigate how sequence length impacts representation quality across model architectures.
- **Workflow**:
  1. Uses the 100 curated yeast genes from `Yeast Gene Selection/100 Genes/`, spanning uniform lengths from 50 bp to 5,000 bp in steps of 50 bp.
  2. `sequence-alignment.ipynb`: Computes pairwise Needleman-Wunsch alignment matrices for each gene.
  3. `glm-inference.ipynb`: Extracts embeddings across all gLMs.
  4. `correlator.ipynb`: Computes correlations for all 100 genes and plots **correlation magnitude vs. sequence length regression lines** (Figure 4 in the research report).

---

### 3. Ranking Method (Neighborhood Preservation)
Located in `Ranking Method/`:
- Non-parametric evaluation of local manifold preservation:
  - `rankingSequences.py`: For each strain, sorts all other 1,010 strains in descending order of Needleman-Wunsch score.
  - `rankingEmbeddings.py`: Sorts all other 1,010 strains in ascending order of embedding Euclidean distance.
  - Computes rank-order consistency and top-$k$ nearest-neighbor overlap to determine whether local neighborhoods are preserved.

---

## Execution Instructions

### 1. Evaluate Single Gene Embeddings
```bash
# Step 1: Generate pairwise Euclidean distance matrices
python "Embeddings Evaluation/Single Gene Embeddings/Embeddings Alignment/alignment.py"

# Step 2: Compute correlation against sequence alignment
python "Embeddings Evaluation/Single Gene Embeddings/main.py"
```

### 2. Run the 100-Gene Benchmark
Open the Jupyter notebook:
```bash
jupyter notebook "Embeddings Evaluation/100 Genes/correlator.ipynb"
```

### 3. Run the Neighborhood Ranking Benchmark
```bash
python "Embeddings Evaluation/Ranking Method/Sequence Rankings/rankingSequences.py"
python "Embeddings Evaluation/Ranking Method/Embeddings Ranking/rankingEmbeddings.py"
```

---

## Key Scientific Findings

1. **Universality of Negative Correlation**: All evaluated gLMs exhibit strong negative correlations with alignment scores, confirming that foundation pretraining teaches models that similar sequences belong closer together in latent space.
2. **Context-Length Sensitivity**:
   - **HyenaDNA-large-1M** performs best on longer sequences (*YAL026C*, 4,068 bp), showing stable correlation across the entire 50–5,000 bp range.
   - **Nucleotide Transformer (NT)** models exhibit their highest correlation on shorter sequences (*YAL007C*, 648 bp), with correlation strength decaying on sequences exceeding 1,000 bp.
   - **DNABERT-S, GROVER, and GENA-LM** demonstrate balanced performance across lengths.

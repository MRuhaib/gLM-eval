# Sequence Alignment Pipeline

This subfolder provides the **ground-truth sequence similarity calculation engine** for the intrinsic evaluation suite, implementing pairwise global alignment (Needleman-Wunsch algorithm) and Hamming distance metrics across 1,011 *Saccharomyces cerevisiae* strains.

---

## Methodological Overview

To test whether Genome Language Models (gLMs) preserve sequence relationships (**Hypothesis 1**), we require an independent, biologically grounded metric of sequence similarity. We use **dynamic programming-based global pairwise alignment**:

```
Strain i Sequence:  A T G C C G T A A C G T ...
                               │ │ │ │   │ │ │ │ │ │ │
Strain j Sequence:  A T G C -  G T A A A G T ...
                    └─────────────────────┘
                 Global Alignment (Needleman-Wunsch)
                               │
                               ▼
            Ground-Truth Alignment Score s_ij (1011 x 1011)
```

For full-genome windowed evaluations where global alignment on large windows becomes computationally intractable, we use parallelized **Hamming distance alignment** on aligned homologous chromosome segments.

---

## Directory Organization

```
Sequence Alignment/
├── main.py                        # Computes pairwise Needleman-Wunsch score matrices (1,011 x 1,011)
├── fullGenome.py                  # Full-genome segmentation & parallelized Hamming/Needleman alignment
├── hammingFill.py                 # Symmetrization utility for Hamming score matrices
├── fill.py                        # Symmetrization utility for Needleman-Wunsch score matrices
├── normalize.py                   # Length-normalization engine for alignment scores
├── read.py                        # Sanity check script inspecting alignment matrix headers
├── test.py                        # Test runner verifying aligner configurations
├── Hamming Scores/                # Stores full-genome Hamming distance matrices (gitignored)
└── Sequence Alignment Scores/     # Stores pairwise Needleman-Wunsch matrices (gitignored)
    └── Biotite/read.py            # Helper script verifying Biotite matrix outputs
```

---

## Technical Details & Aligners

### 1. Single Gene Alignment ([`main.py`])
- **Aligner Engines**:
  - **Biopython (`Bio.Align.PairwiseAligner`)**: Uses modern global alignment mode with default match score $= 1$, mismatch score $= 0$, and affine gap penalties.
  - **Biotite (`biotite.sequence.align.align_optimal`)**: High-performance C-accelerated alignment using `SubstitutionMatrix.std_nucleotide_matrix()`.
- **Parallelization**:
  - Uses Python's `concurrent.futures.ProcessPoolExecutor` to compute the upper triangular entries of the $1,011 \times 1,011$ matrix in parallel across CPU cores.
- **Output**: Emits `<GENE>_scores.csv` containing the full pairwise score matrix.

### 2. Whole-Genome Window Alignment ([`fullGenome.py`])
- Tiles the 12 Mb genomes of each strain into contiguous chunks corresponding to model context lengths ($L \in \{500, 1000, 16000, 32000, 160000\}$).
- Computes pairwise Hamming distances across corresponding windows:
  $$\text{Hamming}(s_1, s_2) = \frac{1}{L} \sum_{k=1}^L \mathbb{I}(s_1[k] \neq s_2[k])$$

### 3. Matrix Symmetrization & Normalization
- `fill.py` & `hammingFill.py`: Since pairwise alignment is symmetric ($s_{ij} = s_{ji}$), only the upper triangle ($i \le j$) is computed; these scripts copy values to the lower triangle to produce complete symmetric matrices.
- `normalize.py`: Normalizes alignment scores by dividing by sequence length to enable unbiased comparison across genes of different sizes.

---

## Execution Instructions

### Running Pairwise Single Gene Alignment
Edit the target gene in `main.py` (e.g., `gene = "YAL001C"`), then run:
```bash
python "Sequence Alignment/main.py"
```

### Symmetrizing the Output Matrix
```bash
python "Sequence Alignment/fill.py"
```

### Running Whole Genome Hamming Alignment
```bash
python "Sequence Alignment/fullGenome.py"
python "Sequence Alignment/hammingFill.py"
```

---

## Downstream Consumers

The generated `.csv` score matrices produced by this subfolder serve as the ground truth for:
- `Embeddings Evaluation/Single Gene Embeddings/main.py`
- `Embeddings Evaluation/100 Genes/correlator.ipynb`
- `Embeddings Evaluation/Ranking Method/Sequence Rankings/rankingSequences.py`

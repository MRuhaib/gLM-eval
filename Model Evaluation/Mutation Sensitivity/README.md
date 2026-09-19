# Mutation Sensitivity & Codon Degeneracy Framework

This subfolder implements the experimental framework for **Hypothesis 3: Embedding Robustness & Codon Degeneracy**; evaluating how Genome Language Model (gLM) embeddings respond to synthetic nucleotide mutations, and proving that models internally reflect the degenerate structure of the genetic code.

---

## Research Overview & The Codon Degeneracy Hypothesis

Mutations are single-nucleotide substitutions that can alter phenotypic expression or disrupt regulatory mechanisms. In this module, we probe the sensitivity of frozen gLM representations to mutational perturbations.

### Key Scientific Discovery: Non-Coding Vulnerability
Across foundation models (DNABERT, Nucleotide Transformer, GENA-LM, GROVER), **point mutations in non-coding DNA induce significantly larger shifts in embedding space than identical mutation rates in coding DNA** ($p < 10^{-15}$, Kolmogorov-Smirnov two-sample test).

### The Codon Degeneracy Explanation
Protein-coding DNA operates under the **redundancy of the genetic code**: multiple distinct 3-nucleotide codons translate to the same amino acid (synonymous substitutions), especially at the **wobble (3rd) codon position**. Consequently, coding sequences possess built-in biological buffering against point mutations. In contrast, non-coding sequences lack this redundancy; substitutions frequently disrupt transcription factor binding sites, RNA secondary structure, or chromatin accessibility elements.

---

## Directory Organization

```
Mutation Sensitivity/
├── mutater.py                     # Synthetic point mutation generator (5% - 50%, 100 replicates)
├── singleMutation.py              # Single-nucleotide point mutation inference & distance engine
├── inference.py                   # Embedding generator for mutated sequences
├── analysis.py                    # Statistical analysis & boxplot visualizer (Figure 5 in report)
├── dataframeMaker.py              # Compiles sequence metrics & distances into unified DataFrame
├── mutSeqAnalysis.ipynb           # Multi-model aggregate visualization notebook
├── test.ipynb                     # Exploratory analysis of GC skew and model tokenization
├── unpickle_and_save.py           # Serialization utility converting pickled frames to CSV
├── 1000_seqs/                     # Large-scale validation across 1,000 independent sequences
│   └── *_allDistances.csv         # Distance distributions per model (gitignored)
├── GB sequences/                  # Codon Degeneracy Investigation (Genomic Benchmarks)
│   ├── GB_one_codon_mutation.py   # Exhaustive n=1 point mutation experiment (wobble position isolation)
│   ├── GB_n_codon_mutation.py     # n-codon random dose-response experiment
│   ├── GB_inference.py            # Batched mutation inference pipeline
│   ├── codon_mut_analysis.ipynb   # Synonymous vs. missense curves stratified by codon position
│   └── random_mut_analysis.ipynb  # Random mutation dose-response curves
├── Human Sequences/               # Cross-Species Verification on Human GRCh38
│   ├── geneFilter.py              # Filters and samples human CDS and intergenic sequences
│   ├── humanSeqInference.py       # Baseline inference on human sequences
│   ├── humanMutSeqInference.py    # Mutation sensitivity on human sequences
│   └── Embeddings/classifier.ipynb# Random Forest classification & SHAP interpretability
└── Sequences/                     # Baseline sequence selection utilities
    └── Original Sequences/
        ├── Coding/codingGeneSelector.py      # Selects yeast genes matching model context lengths
        └── Non-coding/noncodingGeneSelector.py # Selects S. matsunami conserved non-coding elements
```

---

## Experimental Workflows

### 1. Synthetic Mutation & Boxplot Analysis
- `mutater.py`: Takes a sequence of length $L$ and introduces stochastic point mutations at rates $x \in \{5\%, 10\%, 15\%, \dots, 50\%\}$. For each rate, generates 100 independently mutated sequences to ensure statistical robustness.
- `inference.py`: Passes mutated and baseline sequences through target gLMs, recording Euclidean and Cosine distances.
- `analysis.py`: Generates boxplot distributions of distance vs. mutation percentage and calculates Kolmogorov-Smirnov test statistics comparing coding vs. non-coding distributions.

---

### 2. Codon Degeneracy Experiment (Genomic Benchmarks)
Located in `GB sequences/`:

#### Exhaustive $n=1$ Codon Mutation Pipeline (`GB_one_codon_mutation.py`)
This experiment provides direct causal proof of codon degeneracy:
1. **Reading Frame Analysis**: Evaluates sequences across all three potential Open Reading Frames (frames 0, 1, 2).
2. **Codon Position Screening**: For every complete codon in the ORF, screens positions 1, 2, and 3 (the wobble position).
3. **Consequence Tracking**:
   - Classifies alternative base substitutions into **synonymous** (same amino acid) vs. **missense** (different amino acid). Nonsense (stop codon) mutations are excluded.
4. **Embedding Shift Comparison**:
   - Compares the distribution of Euclidean distances for synonymous vs. missense mutations in coding ORFs vs. non-coding sequences.
   - **Hypothesis Confirmed**: Synonymous substitutions at position 3 (wobble) produce significantly smaller embedding shifts than missense substitutions in true coding ORFs, whereas non-coding sequences show no such difference.

#### Multi-Codon Dose-Response (`GB_n_codon_mutation.py`)
Samples $n$ codons without replacement and plots dose-response curves of distance vs. mutation count for synonymous vs. missense substitutions.

---

### 3. Human Reference Sequence Validation
Located in `Human Sequences/`:
- Verifies that the coding/non-coding mutation asymmetry is not an artifact of yeast or fish sequences.
- Samples 1,000 coding CDS sequences and 1,000 intergenic sequences from human GRCh38 using `geneFilter.py`.
- Evaluates transition ($\text{A} \leftrightarrow \text{G}, \text{C} \leftrightarrow \text{T}$) vs. transversion ($\text{A}/\text{G} \leftrightarrow \text{C}/\text{T}$) ratios to confirm that mutation type does not bias the observed distance distributions.

---

## Execution Instructions

### Running the Synthetic Mutation Pipeline
```bash
# Step 1: Generate mutated sequences
python "Mutation Sensitivity/mutater.py"

# Step 2: Run inference on mutated sequences
python "Mutation Sensitivity/inference.py"

# Step 3: Run statistical analysis and boxplot generation
python "Mutation Sensitivity/analysis.py"
```

### Running the Codon Degeneracy Experiment
```bash
# Exhaustive n=1 point mutation experiment
python "Mutation Sensitivity/GB sequences/GB_one_codon_mutation.py"

# Analyze results in Jupyter
jupyter notebook "Mutation Sensitivity/GB sequences/codon_mut_analysis.ipynb"
```

---

## Summary of Results

1. **Non-Coding Sensitivity**: Non-coding DNA displays significantly higher contextual sensitivity to point mutations across almost all evaluated foundation models.
2. **HyenaDNA Exception**: HyenaDNA models exhibited an atypical pattern, showing slightly higher distances for coding sequences, possibly due to character-level inductive biases and convolutional filters.
3. **Causal Proof of Codon Degeneracy**: The wobble position experiment definitively demonstrates that models have implicitly learned the triplet genetic code and amino acid translation mappings without explicit supervised pretraining.

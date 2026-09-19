# Downstream Evaluation: Yeast Phenotypic Trait Prediction

This subfolder was just a short exploratory test; it was not part of the main methodology. It provides a bridge between **intrinsic representation quality** and **extrinsic biological utility**, evaluating whether un-fine-tuned Genome Language Model (gLM) embeddings capture quantitative organism-level phenotypes across 1,011 *Saccharomyces cerevisiae* isolates.

---

## Biological Context & Motivation

While intrinsic evaluation measures geometric and contextual properties in isolation, an important question is whether these un-supervised embeddings retain predictive signal for **empirical organismal traits**.

In the 1,011 yeast genomes project ([Peter et al., 2018](https://doi.org/10.1038/s41586-018-0030-5)), every strain was phenotyped across a wide spectrum of environmental conditions, carbon sources, and chemical stress agents. This module investigates whether frozen embeddings derived from individual or aggregated yeast genes can predict these phenotypic responses using lightweight regression models without end-to-end model fine-tuning.

---

## Directory Organization

```
Downstream Evaluation/
├── phenotypeEval.ipynb            # Notebook training regressors to predict growth phenotypes
└── pheno_matrix.txt               # Matrix of quantitative growth traits across 1,011 strains
```

---

## File Details

### 1. `phenotypeEval.ipynb`
- **Feature Extraction Strategies**:
  - **Single Gene Embeddings**: Probing representations of specific stress-response genes across all 1,011 strains.
  - **Averaged Embeddings**: Computing the mean embedding vector across multiple representative genes per strain.
  - **Concatenated Embeddings**: Stacking multi-gene representations to capture multi-locus variation.
- **Target Phenotypes (from `pheno_matrix.txt`)**:
  - `YPDBENOMYL200`: Growth under 200 mg/L benomyl (an anti-microtubule fungicide that destabilizes the cytoskeleton).
  - `YPACETATE`: Growth on acetate as an alternative non-fermentable carbon source.
  - Additional stress conditions (heavy metals, osmotic stress, temperature extremes).
- **Modeling & Validation**:
  - Evaluates Ridge Regression, Lasso, and Random Forest Regressors.
  - Performs 5-fold cross-validation (`cross_val_score`) measuring $R^2$ variance explained and Mean Squared Error (MSE).

### 2. `pheno_matrix.txt`
- Tab-delimited table containing normalized growth fitness measurements for 1,011 yeast isolates (rows: strain IDs like `AAA`, `AAB`, ..., columns: environmental/chemical assays).

---

## Execution Instructions

Launch the Jupyter notebook:
```bash
jupyter notebook "Downstream Evaluation/phenotypeEval.ipynb"
```

The notebook automatically loads `pheno_matrix.txt`, aligns strain IDs with precomputed embedding files from `Model Inference/Embeddings/`, and outputs cross-validation performance metrics.

---

## Key Takeaways

1. **Phenotypic Signal in Unsupervised Representations**: Even without fine-tuning, embeddings derived from relevant coding genes demonstrate statistically significant predictive power for specific chemical sensitivities (such as benomyl resistance).
2. **Context vs. Noise**: Aggregating embeddings across multiple genes generally stabilizes regression performance compared to single-gene vectors, providing a baseline for intrinsic-to-extrinsic representation benchmarking.

# Model Inference Engine

This subfolder contains the unified **feature extraction and embedding generation pipeline** for running inference across 10+ Genome Language Models (gLMs) on gene sequences and segmented whole genomes.

---

## Directory Organization

```
Model Inference/
├── main.py                        # Single gene inference pipeline across 1,011 yeast strains
├── fullGenome.py                  # Whole-genome segmentation & windowed inference pipeline
├── Full/
│   └── toNPZ.py                   # Compresses raw text embeddings into binary NumPy .npz files
├── Genes/                         # Input FASTA files for canonical genes across 1,011 strains
└── Embeddings/                    # Output directory for generated embeddings (gitignored)
```

---

## Supported Models Matrix

| Model Identifier | Hugging Face Hub Path | Architecture | Embedding Dim ($D$) | Context Limit |
|:---|:---|:---:|:---:|:---:|
| **DNABERT-2** | `zhihan1996/DNABERT-2-117M` | Transformer (BERT) | 768 | 512 bp |
| **DNABERT-S** | `zhihan1996/DNABERT-S` | Transformer (BERT) | 768 | 512 bp |
| **GENA-LM** | `AIRI-Institute/gena-lm-bigbird-base-t2t` | BigBird Sparse Attention | 768 | 36,000 bp |
| **GPN** | `songlab/gpn-brassicales` | CNN-Transformer Hybrid | 512 | 512 bp |
| **GROVER** | `PoetschLab/GROVER` | Transformer (BERT) | 768 | 510 bp |
| **NT-500M** | `InstaDeepAI/nucleotide-transformer-500m-human-ref` | Transformer (BERT) | 1,024 | 1,000 bp |
| **NT-2.5B-MS** | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` | Transformer (BERT) | 2,560 | 1,000 bp |
| **NT-2.5B-1000G** | `InstaDeepAI/nucleotide-transformer-2.5b-1000g` | Transformer (BERT) | 2,560 | 1,000 bp |
| **Hyena-1k** | `LongSafari/hyenadna-tiny-1k-seqlen-hf` | Hyena Operator | 128 / 256 | 1,000 bp |
| **Hyena-16k** | `LongSafari/hyenadna-tiny-16k-seqlen-d128-hf` | Hyena Operator | 128 | 16,000 bp |
| **Hyena-32k** | `LongSafari/hyenadna-small-32k-seqlen-hf` | Hyena Operator | 256 | 32,000 bp |
| **Hyena-160k** | `LongSafari/hyenadna-medium-160k-seqlen-hf` | Hyena Operator | 256 | 160,000 bp |
| **Hyena-450k** | `LongSafari/hyenadna-medium-450k-seqlen-hf` | Hyena Operator | 256 | 450,000 bp |
| **Hyena-1M** | `LongSafari/hyenadna-large-1m-seqlen-hf` | Hyena Operator | 256 | 1,000,000 bp |

---

## Script Documentation

### 1. Single Gene Inference: `main.py`
Iterates through all 1,011 yeast strain sequences for a specified target gene, generates model embeddings, and outputs text files containing raw arrays:
- **Pooling**: Calculates the mean of the hidden states across the sequence length dimension (ignoring special tokens `[CLS]`, `[SEP]`, `[PAD]`).
- **Device Management**: Automatically routes execution to NVIDIA CUDA if available, falling back to CPU.
- **Handling Model Quirks**:
  - Automatically handles custom configuration classes for DNABERT-2 (`BertConfig`).
  - Correctly loads GPN through `gpn.model`.

### 2. Whole Genome Windowed Inference: `fullGenome.py`
Processes complete 12 Mb yeast chromosomes by tiling them into contiguous non-overlapping windows matching each model's maximum context length ($L_{\text{max}}$).

### 3. Compression Utility: `Full/toNPZ.py`
Converts raw string representations from text outputs into compressed binary `.npz` files, reducing disk footprint by $>80\%$ and speeding up downstream loading into NumPy arrays.

---

## Execution Instructions

### Generating Single Gene Embeddings
Edit the target gene or models list in `main.py` if needed, then run:
```bash
python "Model Inference/main.py"
```
Generated embeddings will be written to `Model Inference/Embeddings/<GENE>/`.

### Generating Full Genome Segment Embeddings
```bash
python "Model Inference/fullGenome.py"
```

### Compressing Embeddings to Binary NPZ
```bash
python "Model Inference/Full/toNPZ.py"
```

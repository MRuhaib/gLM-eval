#!/usr/bin/env python3
"""
variant_umap.py
───────────────────────────────────────────────────────────────────────────────
UMAP visualization of raw genotype-variant matrices derived from VCF files
for 1 011 yeast strains across multiple genes.

PURPOSE  (control experiment)
─────────────────────────────
Genome language model (gLM) embeddings of gene sequences cluster by
phylogenetic clade in UMAP space.  This script asks whether the **raw
nucleotide-variant pattern alone** already explains that clustering.

  • If the variant-matrix UMAP shows strong clade separation
        ⟹  gLM clustering could simply reflect nucleotide differences.
  • If the variant-matrix UMAP shows weak / absent clade separation
        ⟹  gLMs likely capture deeper evolutionary or functional signals
            beyond raw variant patterns.

NOTE:  Some degree of clade structure in the variant matrix is expected
because phylogenetic clades are *defined* by shared derived mutations.
The key comparison is the *degree* of clustering relative to the gLM
embeddings — not merely its presence or absence.

APPROACH
────────
  1. Parse each gene's VCF → genotype matrix  (1011 strains × V variant sites)
  2. Binarize genotypes:  REF = 0,  any ALT = 1  (configurable)
  3. Impute missing genotypes (.) with the per-site mode
  4. Reduce to 2-D with UMAP  (same hyper-params as the embedding UMAP)
  5. Scatter-plot colored by phylogenetic clade

Outputs are saved to  Variants Analysis / Results /

Usage:
    python variant_umap.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
VCF_DIR = SCRIPT_DIR / "VCF"
DATA_DIR = SCRIPT_DIR.parent / "Data Labelling"
OUTPUT_DIR = SCRIPT_DIR / "Results"

GENES = [
    "YAL001C",
    "YAL007C",
    "YAL018C",
    "YAL022C",
    "YAL026C",
    "YJR136C",
    "YPR192W",
]

# UMAP hyper-parameters — deliberately matching vizualise.py for fair comparison
UMAP_PARAMS = dict(
    n_neighbors=17,
    min_dist=5,
    spread=10,
    random_state=42,
    # metric defaults to "euclidean"; Hamming is an alternative for binary data
)

# Set True to collapse all ALT alleles into a single 1  (REF=0, ALT=1)
# Set False to keep multi-allelic genotype indices (0, 1, 2, …)
BINARIZE = True

# ── Stable 30-clade colour palette ───────────────────────────────────────────

CLADE_ORDER = [
    "M3",
    "M2",
    "Wine/European",
    "Far East Asia",
    "Mosaic beer",
    "African palm wine",
    "UN",
    "Alpechin",
    "Malaysian",
    "Asian islands",
    "Asian fermentation",
    "M1",
    "North American oak",
    "Mixed origin",
    "Brazilian bioethanol",
    "West African cocoa",
    "Sake",
    "African beer",
    "French Guiana human",
    "Ale beer",
    "Ecuadorean",
    "French dairy",
    "Mexican agave",
    "Taiwanese",
    "CHNI",
    "Mediterranean oak",
    "CHNII",
    "CHN V",
    "CHNIII",
    "Far East Russian",
]


def _build_palette():
    """Combine tab20 + tab20b to get ≥30 visually distinct colours."""
    colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    return {clade: colors[i] for i, clade in enumerate(CLADE_ORDER)}


CLADE_PALETTE = _build_palette()

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_vcf(path: Path):
    """
    Parse a VCF file.

    Returns
    -------
    sample_ids : list[str]
        Column header names for each strain.
    matrix : np.ndarray, shape (n_strains, n_variant_sites)
        Genotype values (0, 1, 2, … or NaN for missing).
    positions : list[int]
        1-based genomic positions of each variant site.
    """
    sample_ids, rows, positions = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_ids = line.strip().split("\t")[9:]
                continue
            parts = line.strip().split("\t")
            positions.append(int(parts[1]))
            rows.append([np.nan if g == "." else int(g) for g in parts[9:]])
    # rows is (n_variants, n_strains) → transpose to (n_strains, n_variants)
    return sample_ids, np.asarray(rows, dtype=float).T, positions


def extract_strain_name(sample_id: str, gene: str) -> str:
    """
    Strip the gene suffix from a VCF sample column name.

    E.g. 'SACE_YAU_YAL001C_TFC3' → 'SACE_YAU'
         'BCS_YAL001C_TFC3'       → 'BCS'
    """
    marker = f"_{gene}_"
    idx = sample_id.find(marker)
    return sample_id[:idx] if idx >= 0 else sample_id


def impute_mode(mat: np.ndarray) -> np.ndarray:
    """Replace NaN with the column (per-variant-site) mode."""
    out = mat.copy()
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mask = np.isnan(col)
        if mask.any():
            valid = col[~mask]
            if len(valid):
                out[mask, j] = stats.mode(valid, keepdims=True).mode[0]
    return out


def compute_clustering_metrics(embedding_2d, clade_labels):
    """
    Compute silhouette score (on the true clade labels) and
    Adjusted Rand Index (KMeans k=30 vs true labels) for the 2-D embedding.
    """
    le = LabelEncoder()
    int_labels = le.fit_transform(clade_labels)
    n_unique = len(set(int_labels))

    sil = silhouette_score(embedding_2d, int_labels) if n_unique >= 2 else float("nan")

    # ARI: cluster the 2-D UMAP output with KMeans and compare to true labels
    km = KMeans(n_clusters=min(n_unique, 30), random_state=42, n_init=10)
    pred = km.fit_predict(embedding_2d)
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(int_labels, pred)

    return sil, ari


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load clade labels
    meta = pd.read_csv(DATA_DIR / "metadata2.csv")
    clade_map = dict(zip(meta["Standardized name"], meta["Clades"].str.strip()))

    # ── Combined subplot grid ─────────────────────────────────────────────
    ncols = min(4, len(GENES))
    nrows = -(-len(GENES) // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols + 3, 5 * nrows),
        squeeze=False,
    )

    metrics_rows = []

    for idx, gene in enumerate(GENES):
        vcf_path = VCF_DIR / f"{gene}.vcf"
        if not vcf_path.exists():
            print(f"  [!] {vcf_path.name} not found — skipping")
            continue

        print(f"\n{'─' * 55}")
        print(f"  Gene: {gene}")
        sample_ids, geno_mat, positions = parse_vcf(vcf_path)
        names = [extract_strain_name(s, gene) for s in sample_ids]
        clades = [clade_map.get(n, "Unknown") for n in names]

        n_strains, n_vars = geno_mat.shape
        n_missing = int(np.isnan(geno_mat).sum())
        n_unknown = sum(1 for c in clades if c == "Unknown")
        print(f"  Strains : {n_strains}")
        print(f"  Variant sites : {n_vars}")
        print(
            f"  Missing GTs   : {n_missing}  ({100 * n_missing / geno_mat.size:.2f}%)"
        )
        if n_unknown:
            print(f"  [!] {n_unknown} strains could not be mapped to a clade")

        # Binarize (optional)
        if BINARIZE:
            geno_mat = np.where(
                np.isnan(geno_mat), np.nan, np.where(geno_mat == 0, 0.0, 1.0)
            )

        # Impute missing values
        geno_mat = impute_mode(geno_mat)

        # UMAP
        print(f"  Running UMAP …")
        reducer = umap.UMAP(**UMAP_PARAMS)
        emb = reducer.fit_transform(geno_mat)

        df = pd.DataFrame(
            {
                "UMAP1": emb[:, 0],
                "UMAP2": emb[:, 1],
                "Clade": clades,
            }
        )

        # Metrics
        sil, ari = compute_clustering_metrics(emb, clades)
        print(f"  Silhouette score (true clades) : {sil:.4f}")
        print(f"  Adjusted Rand Index (KMeans)   : {ari:.4f}")
        metrics_rows.append(
            {
                "Gene": gene,
                "Variants": n_vars,
                "Silhouette": round(sil, 4),
                "ARI": round(ari, 4),
            }
        )

        # ── Individual gene figure ────────────────────────────────────────
        fig_i, ax_i = plt.subplots(figsize=(9, 7))
        sns.scatterplot(
            data=df,
            x="UMAP1",
            y="UMAP2",
            hue="Clade",
            palette=CLADE_PALETTE,
            s=15,
            ax=ax_i,
        )
        mode_label = "Binary REF/ALT" if BINARIZE else "Multi-allelic GT"
        ax_i.set_title(
            f"UMAP of Variant Matrix — {gene}  "
            f"({n_vars} variant sites, {mode_label})",
            fontsize=13,
        )
        ax_i.legend(
            title="Clade",
            fontsize=7,
            title_fontsize=9,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=True,
        )
        ax_i.set_aspect("equal", adjustable="box")
        ax_i.grid(True, alpha=0.3, linewidth=0.5)
        for sp in ax_i.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("black")
        fig_i.tight_layout()
        fig_i.savefig(
            OUTPUT_DIR / f"variant_umap_{gene}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_i)
        print(f"  → saved variant_umap_{gene}.png")

        # ── Combined subplot panel ────────────────────────────────────────
        ax = axes[idx // ncols, idx % ncols]
        sns.scatterplot(
            data=df,
            x="UMAP1",
            y="UMAP2",
            hue="Clade",
            palette=CLADE_PALETTE,
            s=8,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{gene}  ({n_vars} vars)", fontsize=11, pad=6)
        ax.set_xlabel("UMAP1", fontsize=9)
        ax.set_ylabel("UMAP2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("black")

    # Hide unused panels
    for i in range(len(GENES), nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    # Shared legend on the right
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markersize=6,
            markerfacecolor=CLADE_PALETTE.get(c, "grey"),
            label=c,
        )
        for c in CLADE_ORDER
    ]
    fig.legend(
        handles=handles,
        title="Clade",
        fontsize=8,
        title_fontsize=10,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        frameon=True,
    )

    mode_str = "Binary (REF / ALT)" if BINARIZE else "Multi-allelic GT indices"
    fig.suptitle(
        f"UMAP of Raw Variant Matrices — {mode_str}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    combined_path = OUTPUT_DIR / "variant_umap_all_genes.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{'═' * 55}")
    print(f"  Combined figure → {combined_path}")

    # ── Print metrics summary ─────────────────────────────────────────────
    if metrics_rows:
        print(f"\n  Clustering Metrics Summary")
        print(f"  {'─' * 50}")
        metrics_df = pd.DataFrame(metrics_rows)
        print(metrics_df.to_string(index=False))
        metrics_df.to_csv(OUTPUT_DIR / "variant_umap_metrics.csv", index=False)
        print(f"\n  Metrics saved → {OUTPUT_DIR / 'variant_umap_metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
glm_variant_comparison.py

Compare raw-variant UMAP structure against gLM embedding UMAP structure
for all available model embeddings and all target genes.

Outputs are written to:
  Variants Analysis/Results/glm_variant_comparison/
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).parent
VCF_DIR = SCRIPT_DIR / "VCF"
DATA_DIR = SCRIPT_DIR.parent / "Data Labelling"
EMBED_DIR = SCRIPT_DIR.parent.parent / "Model Inference" / "Embeddings"
OUTPUT_DIR = SCRIPT_DIR / "Results" / "glm_variant_comparison"

GENES = [
    "YAL001C",
    "YAL007C",
    "YAL018C",
    "YAL022C",
    "YAL026C",
    "YJR136C",
    "YPR192W",
]

UMAP_PARAMS = dict(
    n_neighbors=17,
    min_dist=5,
    spread=10,
    random_state=42,
)

BINARIZE_VARIANTS = True
TOP_K_LIST = [1, 5, 10]


def parse_vcf(path: Path) -> Tuple[List[str], np.ndarray]:
    sample_ids, rows = [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_ids = line.rstrip("\n").split("\t")[9:]
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append([np.nan if g == "." else float(g) for g in parts[9:]])
    if not rows:
        raise ValueError(f"No variant rows found in {path}")
    mat = np.asarray(rows, dtype=float).T
    return sample_ids, mat


def extract_strain_name(sample_id: str, gene: str) -> str:
    marker = f"_{gene}_"
    idx = sample_id.find(marker)
    return sample_id[:idx] if idx >= 0 else sample_id


def impute_mode(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isnan(col)
        if mask.any():
            valid = col[~mask]
            if len(valid):
                out[mask, j] = stats.mode(valid, keepdims=True).mode[0]
            else:
                out[mask, j] = 0.0
    return out


def discover_embedding_files(gene: str) -> List[Path]:
    gene_dir = EMBED_DIR / gene
    if not gene_dir.exists():
        return []
    files = sorted(gene_dir.rglob(f"*_{gene}.txt"))
    return files


def model_name_from_file(path: Path, gene: str) -> str:
    rel = path.relative_to(EMBED_DIR / gene)
    stem = rel.stem
    suffix = f"_{gene}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    if rel.parent == Path("."):
        return stem
    return f"{rel.parent.as_posix()}/{stem}"


def load_embedding_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    obj = ast.literal_eval(text)
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"Unexpected embedding structure in {path}")

    if not isinstance(obj[0], dict) or "embedding" not in obj[0]:
        raise ValueError(f"Expected list of dicts with 'embedding' in {path}")

    has_id = "id" in obj[0]
    if has_id:
        ids = [int(item["id"]) for item in obj]
        if ids == list(range(1, len(ids) + 1)):
            obj = sorted(obj, key=lambda x: int(x["id"]))

    emb = np.asarray([item["embedding"] for item in obj], dtype=float)
    return emb


def clustering_metrics(coords: np.ndarray, clades: List[str]) -> Tuple[float, float]:
    le = LabelEncoder()
    labels = le.fit_transform(clades)
    n_unique = len(np.unique(labels))

    sil = silhouette_score(coords, labels) if n_unique >= 2 else np.nan
    km = KMeans(n_clusters=min(n_unique, 30), random_state=42, n_init=10)
    pred = km.fit_predict(coords)
    ari_true = adjusted_rand_score(labels, pred)
    return float(sil), float(ari_true)


def kmeans_labels(coords: np.ndarray, k: int = 30) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(coords)


def knn_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    nn_a = NearestNeighbors(n_neighbors=k + 1).fit(a)
    nn_b = NearestNeighbors(n_neighbors=k + 1).fit(b)
    idx_a = nn_a.kneighbors(return_distance=False)[:, 1:]
    idx_b = nn_b.kneighbors(return_distance=False)[:, 1:]

    scores = []
    for i in range(a.shape[0]):
        sa = set(idx_a[i].tolist())
        sb = set(idx_b[i].tolist())
        scores.append(len(sa & sb) / float(k))
    return float(np.mean(scores))


def orthogonal_align_to_reference(
    ref_coords: np.ndarray, query_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ref_centered = ref_coords - ref_coords.mean(axis=0, keepdims=True)
    qry_centered = query_coords - query_coords.mean(axis=0, keepdims=True)

    ref_norm = np.linalg.norm(ref_centered)
    qry_norm = np.linalg.norm(qry_centered)
    if ref_norm == 0 or qry_norm == 0:
        return ref_centered, qry_centered

    ref_scaled = ref_centered / ref_norm
    qry_scaled = qry_centered / qry_norm

    r, _ = orthogonal_procrustes(qry_scaled, ref_scaled)
    aligned = qry_scaled @ r
    return ref_scaled, aligned


def pairwise_distance_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    da = pdist(a)
    db = pdist(b)
    pearson = float(np.corrcoef(da, db)[0, 1])
    spearman = float(stats.spearmanr(da, db).correlation)
    return pearson, spearman


def retrieval_metrics(
    ref_coords: np.ndarray, aligned_query_coords: np.ndarray, top_k: List[int]
) -> Dict[str, float]:
    dist = squareform(pdist(np.vstack([ref_coords, aligned_query_coords])))
    n = ref_coords.shape[0]
    cross = dist[:n, n:]

    diag = np.diag(cross)
    out = {
        "diag_mean_distance": float(np.mean(diag)),
        "diag_median_distance": float(np.median(diag)),
    }

    order = np.argsort(cross, axis=1)
    for k in top_k:
        hits = 0
        for i in range(n):
            if i in order[i, :k]:
                hits += 1
        out[f"retrieval_top{k}"] = float(hits / n)

    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(DATA_DIR / "metadata2.csv")
    clade_map = dict(zip(metadata["Standardized name"], metadata["Clades"].str.strip()))

    variant_cache: Dict[str, Dict[str, object]] = {}
    results_rows = []
    gene_rows = []

    print("Preparing variant-space references per gene...")
    for gene in GENES:
        vcf_path = VCF_DIR / f"{gene}.vcf"
        if not vcf_path.exists():
            print(f"  [skip] Missing VCF: {vcf_path.name}")
            continue

        sample_ids, var_mat = parse_vcf(vcf_path)
        if BINARIZE_VARIANTS:
            var_mat = np.where(np.isnan(var_mat), np.nan, np.where(var_mat == 0, 0.0, 1.0))
        var_mat = impute_mode(var_mat)

        strain_names = [extract_strain_name(sid, gene) for sid in sample_ids]
        clades = [clade_map.get(name, "Unknown") for name in strain_names]

        reducer = umap.UMAP(**UMAP_PARAMS)
        variant_umap = reducer.fit_transform(var_mat)

        var_sil, var_ari_true = clustering_metrics(variant_umap, clades)
        var_kmeans = kmeans_labels(variant_umap, k=30)

        variant_cache[gene] = {
            "coords": variant_umap,
            "kmeans": var_kmeans,
            "clades": clades,
            "silhouette_true_clades": var_sil,
            "ari_kmeans_vs_true_clades": var_ari_true,
            "n_samples": int(var_mat.shape[0]),
            "n_variant_features": int(var_mat.shape[1]),
        }

        gene_rows.append(
            {
                "gene": gene,
                "space": "variant",
                "model": "variant_reference",
                "n_samples": int(var_mat.shape[0]),
                "n_features": int(var_mat.shape[1]),
                "silhouette_true_clades": var_sil,
                "ari_kmeans_vs_true_clades": var_ari_true,
            }
        )

    print("Running gLM-vs-variant comparisons...")
    for gene in sorted(variant_cache.keys()):
        ref = variant_cache[gene]
        ref_coords = ref["coords"]
        ref_kmeans = ref["kmeans"]
        clades = ref["clades"]

        emb_files = discover_embedding_files(gene)
        print(f"\nGene {gene}: {len(emb_files)} embedding file(s)")

        for emb_file in emb_files:
            model = model_name_from_file(emb_file, gene)
            try:
                emb = load_embedding_matrix(emb_file)
            except Exception as exc:
                print(f"  [skip] {model}: parse error ({exc})")
                continue

            if emb.shape[0] != ref_coords.shape[0]:
                print(f"  [skip] {model}: sample count mismatch {emb.shape[0]} vs {ref_coords.shape[0]}")
                continue

            glm_coords = umap.UMAP(**UMAP_PARAMS).fit_transform(emb)

            glm_sil, glm_ari_true = clustering_metrics(glm_coords, clades)
            glm_kmeans = kmeans_labels(glm_coords, k=30)

            ari_variant_glm = adjusted_rand_score(ref_kmeans, glm_kmeans)
            nmi_variant_glm = normalized_mutual_info_score(ref_kmeans, glm_kmeans)

            pearson_dist_corr, spearman_dist_corr = pairwise_distance_correlation(ref_coords, glm_coords)
            knn_overlap_10 = knn_overlap(ref_coords, glm_coords, k=10)

            ref_aligned, aligned_glm = orthogonal_align_to_reference(ref_coords, glm_coords)
            disparity = float(np.mean(np.sum((ref_aligned - aligned_glm) ** 2, axis=1)))

            retrieval = retrieval_metrics(ref_aligned, aligned_glm, TOP_K_LIST)

            row = {
                "gene": gene,
                "model": model,
                "embedding_file": str(emb_file),
                "n_samples": int(emb.shape[0]),
                "n_features": int(emb.shape[1]),
                "variant_silhouette_true_clades": float(ref["silhouette_true_clades"]),
                "variant_ari_kmeans_vs_true_clades": float(ref["ari_kmeans_vs_true_clades"]),
                "glm_silhouette_true_clades": glm_sil,
                "glm_ari_kmeans_vs_true_clades": glm_ari_true,
                "ari_variant_vs_glm_kmeans": float(ari_variant_glm),
                "nmi_variant_vs_glm_kmeans": float(nmi_variant_glm),
                "pairwise_dist_pearson": pearson_dist_corr,
                "pairwise_dist_spearman": spearman_dist_corr,
                "knn_overlap_at_10": float(knn_overlap_10),
                "procrustes_like_mse": disparity,
            }
            row.update(retrieval)
            results_rows.append(row)

            gene_rows.append(
                {
                    "gene": gene,
                    "space": "glm",
                    "model": model,
                    "n_samples": int(emb.shape[0]),
                    "n_features": int(emb.shape[1]),
                    "silhouette_true_clades": glm_sil,
                    "ari_kmeans_vs_true_clades": glm_ari_true,
                }
            )

    if not results_rows:
        raise RuntimeError("No comparisons were produced. Check input files.")

    comparisons = pd.DataFrame(results_rows)
    gene_level = pd.DataFrame(gene_rows)

    comparisons_path = OUTPUT_DIR / "glm_vs_variant_comparison_metrics.csv"
    gene_level_path = OUTPUT_DIR / "gene_level_clustering_metrics.csv"
    comparisons.to_csv(comparisons_path, index=False)
    gene_level.to_csv(gene_level_path, index=False)

    model_summary = (
        comparisons.groupby("model", as_index=False)
        .agg(
            n_genes=("gene", "count"),
            mean_glm_silhouette=("glm_silhouette_true_clades", "mean"),
            mean_glm_ari_true=("glm_ari_kmeans_vs_true_clades", "mean"),
            mean_ari_variant_glm=("ari_variant_vs_glm_kmeans", "mean"),
            mean_nmi_variant_glm=("nmi_variant_vs_glm_kmeans", "mean"),
            mean_pairwise_dist_spearman=("pairwise_dist_spearman", "mean"),
            mean_knn_overlap_10=("knn_overlap_at_10", "mean"),
            mean_procrustes_like_mse=("procrustes_like_mse", "mean"),
            mean_diag_distance=("diag_mean_distance", "mean"),
            mean_retrieval_top1=("retrieval_top1", "mean"),
            mean_retrieval_top5=("retrieval_top5", "mean"),
            mean_retrieval_top10=("retrieval_top10", "mean"),
        )
        .sort_values(by="mean_nmi_variant_glm", ascending=False)
    )
    summary_path = OUTPUT_DIR / "model_summary_metrics.csv"
    model_summary.to_csv(summary_path, index=False)

    pivot_nmi = comparisons.pivot(index="model", columns="gene", values="nmi_variant_vs_glm_kmeans")
    plt.figure(figsize=(10, max(6, 0.35 * len(pivot_nmi))))
    sns.heatmap(pivot_nmi, cmap="viridis", annot=True, fmt=".2f", cbar_kws={"label": "NMI"})
    plt.title("NMI between Variant-UMAP and gLM-UMAP KMeans Clusters")
    plt.xlabel("Gene")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_nmi_variant_vs_glm.png", dpi=300)
    plt.close()

    pivot_top1 = comparisons.pivot(index="model", columns="gene", values="retrieval_top1")
    plt.figure(figsize=(10, max(6, 0.35 * len(pivot_top1))))
    sns.heatmap(pivot_top1, cmap="mako", annot=True, fmt=".2f", cbar_kws={"label": "Top-1"})
    plt.title("Per-Strain Correspondence: Top-1 Retrieval after Alignment")
    plt.xlabel("Gene")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_top1_retrieval.png", dpi=300)
    plt.close()

    print("\nSaved:")
    print(f"  {comparisons_path}")
    print(f"  {gene_level_path}")
    print(f"  {summary_path}")
    print(f"  {OUTPUT_DIR / 'heatmap_nmi_variant_vs_glm.png'}")
    print(f"  {OUTPUT_DIR / 'heatmap_top1_retrieval.png'}")


if __name__ == "__main__":
    main()

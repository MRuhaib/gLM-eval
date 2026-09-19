#!/usr/bin/env python3
"""
Build per-gene SNP one-hot encodings from VCF files.

For each gene VCF, creates a matrix with shape:
    (n_strains, 4 * n_selected_sites)
where each site contributes a 4-wide block in A,C,G,T order.

Outputs are written under:
    Variants Analysis/SNP_OneHot/<GENE>/

Files per gene:
- snp_onehot.npy              : uint8 matrix [n_strains, 4*n_sites]
- vcf_gt_matrix.npy           : int16 matrix [n_strains, n_sites] (GT allele index)
- sample_ids.txt              : sample IDs in row order
- sites.csv                   : site metadata (POS, REF, ALT, kept index)
- feature_names.txt           : feature names in column order
- vcf_feature_names.txt       : VCF feature names (one per site)
- vcf_gt_matrix.tsv           : text table with sample_id + raw GT site columns
- snp_onehot_bundle.npz       : matrix + metadata in one file

A run summary is also saved:
- Variants Analysis/SNP_OneHot/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

# IUPAC ambiguity map. Used when ALT/REF contains ambiguity codes.
IUPAC_TO_BASES: Dict[str, Set[str]] = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
    "N": {"A", "C", "G", "T"},
}


def decode_base_symbol(symbol: str) -> Set[str]:
    symbol = symbol.upper().strip()
    return IUPAC_TO_BASES.get(symbol, set())


def parse_gt_token(raw: str) -> List[int]:
    """
    Parse a VCF GT token and return allele indices.

    Supports forms like: "0", "1", "0/1", "1|1", "0:...", ".".
    """
    gt = raw.split(":", 1)[0]
    if gt == ".":
        return []

    gt = gt.replace("|", "/")
    parts = gt.split("/")

    out: List[int] = []
    for p in parts:
        p = p.strip()
        if not p or p == ".":
            continue
        try:
            out.append(int(p))
        except ValueError:
            return []
    return out


def allele_index_to_bases(allele_idx: int, ref: str, alts: List[str]) -> Set[str]:
    if allele_idx == 0:
        return decode_base_symbol(ref)
    alt_i = allele_idx - 1
    if alt_i < 0 or alt_i >= len(alts):
        return set()
    return decode_base_symbol(alts[alt_i])


def site_is_supported(ref: str, alts: List[str], strict_snp_only: bool) -> bool:
    """
    Keep only sites that can be represented in A/C/G/T one-hot blocks.

    strict_snp_only=True: keep only one-character REF/ALT symbols.
    strict_snp_only=False: still requires symbols to decode to A/C/G/T sets.
    """
    if strict_snp_only:
        if len(ref.strip()) != 1:
            return False
        if any(len(a.strip()) != 1 for a in alts):
            return False

    if not decode_base_symbol(ref):
        return False

    for a in alts:
        if not decode_base_symbol(a):
            return False

    return True


def parse_vcf_rows(vcf_path: Path) -> Tuple[List[str], List[List[str]]]:
    sample_ids: List[str] = []
    rows: List[List[str]] = []

    with open(vcf_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_ids = line.rstrip("\n").split("\t")[9:]
                continue
            if not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))

    if not sample_ids:
        raise ValueError(f"No #CHROM header found in {vcf_path}")

    return sample_ids, rows


def build_onehot_from_vcf(vcf_path: Path, strict_snp_only: bool) -> Dict[str, object]:
    sample_ids, rows = parse_vcf_rows(vcf_path)
    n_samples = len(sample_ids)

    kept_sites = []
    col_blocks = []
    gt_cols = []

    for parts in rows:
        pos = parts[1]
        ref = parts[3].upper()
        alts = [a.upper() for a in parts[4].split(",")]
        gt_calls = parts[9:]

        if len(gt_calls) != n_samples:
            continue

        if not site_is_supported(ref, alts, strict_snp_only):
            continue

        # Build one 4-d block per sample for this site.
        # Supports ambiguous calls by setting multiple bits within a site block.
        block = np.zeros((n_samples, 4), dtype=np.uint8)
        gt_col = np.full(n_samples, -1, dtype=np.int16)
        non_ref_seen = False

        for i, gt_raw in enumerate(gt_calls):
            allele_indices = parse_gt_token(gt_raw)
            if not allele_indices:
                continue

            # Store raw GT allele index representation for this site.
            # For multi-token GT strings, we keep the first parsed index.
            gt_col[i] = int(allele_indices[0])

            base_set: Set[str] = set()
            for ai in allele_indices:
                base_set |= allele_index_to_bases(ai, ref, alts)

            if not base_set:
                continue

            # mark presence of any non-reference allele index
            if any(ai > 0 for ai in allele_indices):
                non_ref_seen = True

            for b in base_set:
                block[i, BASE_TO_IDX[b]] = 1

        # Keep site only if there is variation among samples.
        if not non_ref_seen:
            continue

        kept_sites.append({"pos": int(pos), "ref": ref, "alt": ",".join(alts)})
        col_blocks.append(block)
        gt_cols.append(gt_col)

    if not col_blocks:
        matrix = np.zeros((n_samples, 0), dtype=np.uint8)
        feature_names: List[str] = []
        vcf_gt_matrix = np.zeros((n_samples, 0), dtype=np.int16)
        vcf_feature_names: List[str] = []
    else:
        # Convert (sites) list of [n_samples, 4] into [n_samples, 4*n_sites]
        matrix = np.concatenate(col_blocks, axis=1)
        vcf_gt_matrix = np.column_stack(gt_cols)
        feature_names = []
        vcf_feature_names = []
        for s in kept_sites:
            for b in BASES:
                feature_names.append(f"pos{s['pos']}_{b}")
            vcf_feature_names.append(f"pos{s['pos']}")

    return {
        "sample_ids": sample_ids,
        "sites": kept_sites,
        "matrix": matrix,
        "feature_names": feature_names,
        "vcf_gt_matrix": vcf_gt_matrix,
        "vcf_feature_names": vcf_feature_names,
    }


def infer_genes(vcf_dir: Path) -> List[str]:
    return sorted(p.stem for p in vcf_dir.glob("*.vcf"))


def save_gene_outputs(
    gene: str, out_dir: Path, result: Dict[str, object]
) -> Dict[str, object]:
    gene_dir = out_dir / gene
    gene_dir.mkdir(parents=True, exist_ok=True)

    sample_ids: List[str] = result["sample_ids"]
    sites: List[Dict[str, object]] = result["sites"]
    matrix: np.ndarray = result["matrix"]
    feature_names: List[str] = result["feature_names"]
    vcf_gt_matrix: np.ndarray = result["vcf_gt_matrix"]
    vcf_feature_names: List[str] = result["vcf_feature_names"]

    np.save(gene_dir / "snp_onehot.npy", matrix)
    np.save(gene_dir / "vcf_gt_matrix.npy", vcf_gt_matrix)

    with open(gene_dir / "sample_ids.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(sample_ids))
        fh.write("\n")

    pd.DataFrame(sites).to_csv(gene_dir / "sites.csv", index=False)

    with open(gene_dir / "feature_names.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(feature_names))
        fh.write("\n")

    with open(gene_dir / "vcf_feature_names.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(vcf_feature_names))
        fh.write("\n")

    vcf_df = pd.DataFrame(vcf_gt_matrix, columns=vcf_feature_names)
    vcf_df.insert(0, "sample_id", sample_ids)
    vcf_df.to_csv(gene_dir / "vcf_gt_matrix.tsv", sep="\t", index=False)

    np.savez_compressed(
        gene_dir / "snp_onehot_bundle.npz",
        matrix=matrix,
        vcf_gt_matrix=vcf_gt_matrix,
        sample_ids=np.asarray(sample_ids, dtype=object),
        positions=np.asarray([s["pos"] for s in sites], dtype=int),
        refs=np.asarray([s["ref"] for s in sites], dtype=object),
        alts=np.asarray([s["alt"] for s in sites], dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        vcf_feature_names=np.asarray(vcf_feature_names, dtype=object),
        base_order=np.asarray(BASES, dtype=object),
    )

    return {
        "gene": gene,
        "n_samples": matrix.shape[0],
        "n_sites": len(sites),
        "n_features": matrix.shape[1],
        "n_vcf_features": vcf_gt_matrix.shape[1],
        "density": float(matrix.mean()) if matrix.size else 0.0,
        "output_dir": str(gene_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 4xS SNP one-hot matrices (A/C/G/T per variable site) from VCF files."
    )
    parser.add_argument(
        "--vcf-dir",
        type=Path,
        default=Path(__file__).parent / "VCF",
        help="Directory containing per-gene VCF files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "SNP_OneHot",
        help="Output directory for one-hot matrices.",
    )
    parser.add_argument(
        "--genes",
        nargs="*",
        default=None,
        help="Optional list of genes. If omitted, all *.vcf in --vcf-dir are used.",
    )
    parser.add_argument(
        "--strict-snp-only",
        action="store_true",
        help="Keep only single-symbol REF/ALT sites (drops indels and complex multi-char alleles).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    genes = args.genes if args.genes else infer_genes(args.vcf_dir)
    if not genes:
        raise RuntimeError(f"No genes found in {args.vcf_dir}")

    summary_rows = []

    for gene in genes:
        vcf_path = args.vcf_dir / f"{gene}.vcf"
        if not vcf_path.exists():
            print(f"[skip] Missing VCF: {vcf_path}")
            continue

        print(f"Processing {gene} ...")
        result = build_onehot_from_vcf(vcf_path, strict_snp_only=args.strict_snp_only)
        row = save_gene_outputs(gene, args.out_dir, result)
        summary_rows.append(row)
        print(
            f"  samples={row['n_samples']}  sites={row['n_sites']}  features={row['n_features']}"
        )

    if not summary_rows:
        raise RuntimeError("No output generated. Check input VCF files.")

    summary_df = pd.DataFrame(summary_rows).sort_values("gene")
    summary_path = args.out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

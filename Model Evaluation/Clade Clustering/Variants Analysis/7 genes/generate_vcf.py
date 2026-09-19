#!/usr/bin/env python3
"""
generate_vcf.py
---------------
Generate VCF-like files for each gene FASTA file in the Genes/ directory.

Each FASTA file contains 1011 sequences of a gene (one per yeast strain).
Variants are called relative to the FIRST sequence, which is treated as the
reference. Each other sequence is pairwise-aligned to the reference using
difflib.SequenceMatcher (fast, C-backed, ideal for nearly-identical sequences).

Output: one .vcf file per gene in VCF/

VCF columns:
  CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT  <strain1> <strain2> ...

Genotype (GT) codes:
  0  = REF allele
  1+ = ALT allele index
  .  = deletion relative to reference (gap in alignment)

INFO fields per variant site:
  AC  – allele count for each ALT allele
  AF  – allele frequency for each ALT allele
  NS  – number of strains with a non-missing call
  VAR_TYPE – SNP | DEL | MIXED
"""

import os
import sys
from pathlib import Path
from collections import Counter
from Bio import Align

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
GENES_DIR = SCRIPT_DIR / "Genes"
OUTPUT_DIR = SCRIPT_DIR / "VCF"


# ── Aligner (module-level singleton, reused for all sequences) ────────────────
_aligner = Align.PairwiseAligner()
_aligner.mode = "global"
_aligner.match_score = 2
_aligner.mismatch_score = -1
_aligner.open_gap_score = -4
_aligner.extend_gap_score = -0.5


# ── Alignment ─────────────────────────────────────────────────────────────────


def align_to_reference(ref_seq: str, query_seq: str) -> list[str]:
    """
    Align *query_seq* to *ref_seq* using BioPython's PairwiseAligner (C-backed).

    Returns a list of length len(ref_seq).
    Each element is the nucleotide in the query that aligns to that reference
    position, or '-' if the query has a deletion there.

    Uses the .aligned coordinate blocks (no string formatting), which is fast.
    """
    result: list[str] = ["-"] * len(ref_seq)

    # Get the single best alignment; next() avoids enumerating all equally-scored paths
    aln = next(iter(_aligner.align(ref_seq, query_seq)))

    # .aligned returns (ref_blocks, query_blocks) as arrays of [start, end) pairs
    ref_blocks, qry_blocks = aln.aligned

    for (r_start, r_end), (q_start, q_end) in zip(ref_blocks, qry_blocks):
        # Both spans must be equal length (gapless block)
        for k in range(r_end - r_start):
            result[r_start + k] = query_seq[q_start + k]

    return result


# ── VCF writer ────────────────────────────────────────────────────────────────


def process_fasta(fasta_path: Path, output_dir: Path) -> None:
    """Read one FASTA, align all sequences to the first, write a VCF."""
    from Bio import SeqIO  # import here so the rest of the script works without Bio

    gene_name = fasta_path.stem
    print(f"\n[{gene_name}] Reading sequences …", flush=True)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    n_strains = len(records)

    ref_record = records[0]
    ref_seq = str(ref_record.seq).upper()
    ref_len = len(ref_seq)

    strain_ids = [r.id for r in records]

    # matrix[pos][strain_idx] = nucleotide (or '-' or None)
    print(f"[{gene_name}] Allocating {ref_len} × {n_strains} matrix …", flush=True)
    matrix: list[list[str | None]] = [[None] * n_strains for _ in range(ref_len)]

    # --- Fill reference column (index 0) ---
    for i, nuc in enumerate(ref_seq):
        matrix[i][0] = nuc.upper()

    # --- Align every other strain to reference ---
    print(
        f"[{gene_name}] Aligning {n_strains - 1} sequences to reference …", flush=True
    )
    for idx, record in enumerate(records[1:], start=1):
        query_seq = str(record.seq).upper()
        aligned = align_to_reference(ref_seq, query_seq)
        for pos, nuc in enumerate(aligned):
            matrix[pos][idx] = nuc

        if idx % 200 == 0 or idx == n_strains - 1:
            print(f"  … {idx}/{n_strains - 1}", flush=True)

    # --- Write VCF ---
    output_path = output_dir / f"{gene_name}.vcf"
    variant_count = 0

    print(f"[{gene_name}] Writing {output_path.name} …", flush=True)
    with open(output_path, "w") as fh:
        # Meta-information lines
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##reference={gene_name}_strain0\n")
        fh.write(f"##CHROM={gene_name}\n")
        fh.write(
            '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in called genotypes">\n'
        )
        fh.write(
            '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency among called genotypes">\n'
        )
        fh.write(
            '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of strains with a non-missing call">\n'
        )
        fh.write(
            '##INFO=<ID=VAR_TYPE,Number=1,Type=String,Description="SNP | DEL | MIXED">\n'
        )
        fh.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype (allele index; . = missing/deletion)">\n'
        )

        # Column header
        col_header = [
            "#CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO",
            "FORMAT",
        ] + strain_ids
        fh.write("\t".join(col_header) + "\n")

        # One row per variable site
        for pos in range(ref_len):
            col = matrix[pos]
            ref_nuc = col[0]

            if ref_nuc in (None, "-", "N"):
                continue  # skip gaps/unknown in reference

            # Tally non-missing calls
            calls = [nuc for nuc in col if nuc is not None]
            unique = set(calls) - {"-"}  # ignore deletions for ALT list

            if len(unique) <= 1 and "-" not in calls:
                continue  # monomorphic, no variation

            # Build ALT allele list (sorted, no REF)
            alt_nucs = sorted(unique - {ref_nuc})

            has_del = "-" in calls
            if has_del:
                alt_nucs.append(
                    "*"
                )  # VCF convention: * = spanning deletion / absent allele

            if not alt_nucs:
                continue  # only reference allele observed

            # Allele → index map  (0 = REF)
            allele_idx: dict[str, str] = {ref_nuc: "0"}
            for i, a in enumerate(alt_nucs, start=1):
                allele_idx[a] = str(i)

            # Genotype strings per strain
            gts: list[str] = []
            for nuc in col:
                if nuc is None:
                    gts.append(".")
                elif nuc == "-":
                    gts.append(allele_idx.get("*", "."))
                else:
                    gts.append(allele_idx.get(nuc, "."))

            # INFO
            n_samples_called = sum(1 for g in gts if g != ".")
            ac_values = [gts.count(str(i)) for i in range(1, len(alt_nucs) + 1)]
            af_values = [
                round(ac / n_samples_called, 6) if n_samples_called > 0 else 0.0
                for ac in ac_values
            ]

            # Determine variant type
            alt_display = [a for a in alt_nucs if a != "*"]
            if not alt_display:
                var_type = "DEL"
            elif all(len(a) == 1 and a in "ACGTNacgtn" for a in alt_display):
                var_type = "SNP" if not has_del else "MIXED"
            else:
                var_type = "MIXED"

            info_str = (
                f"AC={','.join(map(str, ac_values))};"
                f"AF={','.join(map(str, af_values))};"
                f"NS={n_samples_called};"
                f"VAR_TYPE={var_type}"
            )

            row = [
                gene_name,
                str(pos + 1),  # 1-based position in reference
                ".",
                ref_nuc,
                ",".join(alt_nucs),
                ".",
                "PASS",
                info_str,
                "GT",
            ] + gts

            fh.write("\t".join(row) + "\n")
            variant_count += 1

    print(
        f"[{gene_name}] Done — {variant_count} variant site(s) written → {output_path}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    fasta_files = sorted(GENES_DIR.glob("*.fasta"))
    if not fasta_files:
        print(f"ERROR: No .fasta files found in {GENES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fasta_files)} FASTA file(s) in {GENES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for fasta_path in fasta_files:
        process_fasta(fasta_path, OUTPUT_DIR)

    print("\n=== All genes processed. ===")


if __name__ == "__main__":
    main()

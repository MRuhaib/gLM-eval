#!/usr/bin/env python3
"""
Verification script to check if the setup is correct for embedding generation.
"""

import os
from pathlib import Path
from Bio import SeqIO


def verify_setup():
    """Verify that the setup is correct for embedding generation."""

    print("🔍 Verifying setup for embedding generation...")

    # Check if 100 Genes directory exists
    genes_dir = Path("100 Genes")
    if not genes_dir.exists():
        print("❌ '100 Genes' directory not found!")
        return False

    print("✅ '100 Genes' directory found")

    # Check FASTA files
    fasta_files = list(genes_dir.glob("*.fasta"))
    print(f"📁 Found {len(fasta_files)} FASTA files")

    if len(fasta_files) == 0:
        print("❌ No FASTA files found in '100 Genes' directory!")
        return False

    # Check a sample file
    sample_file = fasta_files[0]
    print(f"🧬 Checking sample file: {sample_file.name}")

    try:
        sequences = list(SeqIO.parse(sample_file, "fasta"))
        print(f"📊 Sample file contains {len(sequences)} sequences")

        if len(sequences) > 0:
            first_seq = sequences[0]
            print(f"📝 First sequence length: {len(first_seq.seq)} bp")
            print(f"🏷️  First sequence ID: {first_seq.id}")

        if len(sequences) != 1011:
            print(f"⚠️  Expected 1011 sequences, found {len(sequences)}")
        else:
            print("✅ Correct number of sequences (1011) found")

    except Exception as e:
        print(f"❌ Error reading sample file: {e}")
        return False

    # Check disk space (approximate calculation)
    total_sequences = len(fasta_files) * 1011
    print(f"📈 Estimated total sequences to process: {total_sequences:,}")

    # Rough estimate: 768-dim embedding * 4 bytes * 9 models
    estimated_size_gb = (total_sequences * 768 * 4 * 9) / (1024**3)
    print(f"💾 Estimated storage needed: ~{estimated_size_gb:.1f} GB")

    print("\n🎯 Setup verification complete!")
    print("💡 Recommendations:")
    print("   - Start with a small batch size (2-4) if GPU memory is limited")
    print("   - Monitor GPU memory usage during processing")
    print("   - Consider running a test with subset first")

    return True


if __name__ == "__main__":
    verify_setup()

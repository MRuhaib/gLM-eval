import os
import shutil
from pathlib import Path

# Read the finalGenes.txt file
with open("finalGenes.txt", "r") as f:
    finalGenes = eval(f.read())

# Create the "100 Genes" directory if it doesn't exist
output_dir = Path("100 Genes")
output_dir.mkdir(exist_ok=True)

# Source directory for all genes
source_dir = Path("All Genes")

# Counter for successfully copied genes
copied_count = 0
not_found_count = 0
not_found_genes = []

print("Copying selected genes...")

for target_length, gene_info in finalGenes.items():
    gene_name, actual_length, distance = gene_info

    # Skip if no gene was found for this length
    if gene_name is None:
        print(f"No gene found for length {target_length}")
        continue

    # Source file path
    source_file = source_dir / f"{gene_name}.fasta"

    # Check if source file exists
    if not source_file.exists():
        print(f"Warning: File not found - {source_file}")
        not_found_count += 1
        not_found_genes.append(gene_name)
        continue

    # Destination file path with new naming convention
    dest_file = output_dir / f"{gene_name}_{actual_length}.fasta"

    try:
        # Copy the file
        shutil.copy2(source_file, dest_file)
        copied_count += 1
        print(f"Copied: {gene_name} (length {actual_length}) -> {dest_file.name}")
    except Exception as e:
        print(f"Error copying {gene_name}: {e}")

print(f"\nSummary:")
print(f"Successfully copied: {copied_count} genes")
print(f"Not found: {not_found_count} genes")

if not_found_genes:
    print(f"Genes not found: {not_found_genes}")

print(f"\nAll selected genes have been copied to the '100 Genes' folder.")

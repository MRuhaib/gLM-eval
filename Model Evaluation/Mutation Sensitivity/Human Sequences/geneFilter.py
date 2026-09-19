from Bio import SeqIO
import random
import json

directories = ["Coding", "Non-coding", "Human CDS", "Human Genes"]

def filter_sequences(folder, min_length=100, max_length=5000, num_sequences=1000):
    sequences = []
    lengths = {}
    
    input_file = f"{directory}/data/cds.fna" if folder == "Human CDS" else f"{directory}/data/gene.fna"

    # Parse the CDS sequences
    for record in SeqIO.parse(input_file, "fasta"):
        if min_length <= len(record.seq) <= max_length:
            sequences.append(record)
            lengths[record.id] = len(record.seq)
    
    # Randomly select 1000 sequences if more are available
    if len(sequences) > num_sequences:
        sequences = random.sample(sequences, num_sequences)
    
    # Write filtered sequences to output file
    SeqIO.write(sequences, f"{folder}/filtered_{directory}.fasta", "fasta")
    with open(f"{folder}/lengths.json",  "w+") as f:
        json.dump(lengths, f)
    print(f"Selected {len(sequences)} sequences within length range {min_length}-{max_length}")

for directory in directories:
    filter_sequences(directory)


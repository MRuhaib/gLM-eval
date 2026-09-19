import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path
from Bio import SeqIO
import pickle


class EmbeddingDataFrameBuilder:
    def __init__(self, base_dir=".", target_model="GROVER"):
        self.base_dir = Path(base_dir)
        self.embeddings_dir = self.base_dir / "Embeddings"
        self.sequences_dir = self.base_dir / "Sequences"
        self.target_model = target_model

        # Initialize the DataFrame
        self.data = []

    def read_embedding_file(self, file_path):
        """Read embedding from .txt file containing a list"""
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                if content:
                    # Parse the embedding list
                    embedding = ast.literal_eval(content)
                    # If it's a list of lists, take the first one
                    if isinstance(embedding, list) and len(embedding) > 0:
                        if isinstance(embedding[0], list):
                            return np.array(embedding[0])
                        else:
                            return np.array(embedding)
                    return np.array(embedding)
                else:
                    return None
        except Exception as e:
            print(f"Error reading embedding file {file_path}: {e}")
            return None

    def read_sequence_file(self, file_path):
        """Read sequences from .fasta or .txt file"""
        try:
            if file_path.suffix == ".fasta":
                # Read FASTA file
                sequences = []
                for record in SeqIO.parse(file_path, "fasta"):
                    sequences.append(str(record.seq))
                return sequences
            elif file_path.suffix == ".txt":
                # Read text file containing list of sequences
                with open(file_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        sequences = ast.literal_eval(content)
                        return sequences
                    return []
        except Exception as e:
            print(f"Error reading sequence file {file_path}: {e}")
            return []

    def extract_mutation_percentage(self, filename):
        """Extract mutation percentage from filename like YCR038W-A_10.txt"""
        parts = filename.split("_")
        if len(parts) > 1:
            try:
                return int(parts[-1].split(".")[0])
            except:
                return 0
        return 0

    def process_original_data(self):
        """Process original (non-mutated) sequences and embeddings for target model only"""
        print(f"Processing original data for {self.target_model}...")

        original_emb_dir = self.embeddings_dir / "Original"
        original_seq_dir = self.sequences_dir / "Original Sequences"

        for seq_type in ["Coding", "Non-coding"]:
            print(f"  Processing {seq_type} sequences...")

            emb_type_dir = original_emb_dir / seq_type / self.target_model
            seq_type_dir = original_seq_dir / seq_type

            if not emb_type_dir.exists():
                print(
                    f"    Skipping {seq_type} - embedding directory not found: {emb_type_dir}"
                )
                continue

            # Iterate through sequence length percentages
            for length_dir in emb_type_dir.iterdir():
                if not length_dir.is_dir():
                    continue

                length_percent = int(length_dir.name)
                print(f"    Processing length {length_percent}%...")

                # Process embeddings
                for emb_file in length_dir.glob("*.txt"):
                    gene_name = emb_file.stem
                    print(f"      Processing gene: {gene_name}")

                    # Read embedding
                    embedding = self.read_embedding_file(emb_file)
                    if embedding is None:
                        print(f"        Failed to read embedding for {gene_name}")
                        continue

                    # Find corresponding sequence file
                    seq_file_fasta = (
                        seq_type_dir
                        / self.target_model
                        / str(length_percent)
                        / f"{gene_name}.fasta"
                    )
                    seq_file_txt = (
                        seq_type_dir
                        / self.target_model
                        / str(length_percent)
                        / f"{gene_name}.txt"
                    )

                    sequences = []
                    if seq_file_fasta.exists():
                        sequences = self.read_sequence_file(seq_file_fasta)
                        print(f"        Found {len(sequences)} sequences in FASTA file")
                    elif seq_file_txt.exists():
                        sequences = self.read_sequence_file(seq_file_txt)
                        print(f"        Found {len(sequences)} sequences in TXT file")

                    if not sequences:
                        print(f"        Warning: No sequences found for {gene_name}")
                        continue

                    # For original sequences, we'll use the first sequence as representative
                    original_sequence = sequences[0] if sequences else ""

                    # Create row data
                    row_data = {
                        "model": self.target_model,
                        "gene_name": gene_name,
                        "sequence_type": seq_type.lower(),
                        "sequence_length_percent": length_percent,
                        "mutation_percentage": 0,  # Original sequences have no mutations
                        "original_sequence": original_sequence,
                        "mutated_sequence": original_sequence,  # Same as original for non-mutated
                        "sequence": original_sequence,  # Keep the actual sequence
                    }

                    # Add embedding dimensions
                    for i, val in enumerate(embedding):
                        row_data[f"embedding_dim_{i}"] = val

                    self.data.append(row_data)
                    print(f"        Added original data for {gene_name}")

    def process_mutated_data(self):
        """Process mutated sequences and embeddings for target model only"""
        print(f"Processing mutated data for {self.target_model}...")

        mutated_emb_dir = self.embeddings_dir / "Mutated"
        mutated_seq_dir = self.sequences_dir / "Mutated Sequences"
        original_seq_dir = self.sequences_dir / "Original Sequences"

        for seq_type in ["Coding", "Non-coding"]:
            print(f"  Processing {seq_type} sequences...")

            emb_type_dir = mutated_emb_dir / seq_type / self.target_model
            seq_type_dir = mutated_seq_dir / seq_type
            orig_seq_type_dir = original_seq_dir / seq_type

            if not emb_type_dir.exists():
                print(
                    f"    Skipping {seq_type} - embedding directory not found: {emb_type_dir}"
                )
                continue

            # Iterate through sequence length percentages
            for length_dir in emb_type_dir.iterdir():
                if not length_dir.is_dir():
                    continue

                length_percent = int(length_dir.name)
                print(f"    Processing length {length_percent}%...")

                # Process embeddings
                for emb_file in length_dir.glob("*.txt"):
                    file_stem = emb_file.stem
                    mutation_percent = self.extract_mutation_percentage(file_stem)
                    gene_name = file_stem.rsplit("_", 1)[
                        0
                    ]  # Remove mutation percentage

                    print(
                        f"      Processing {gene_name} with {mutation_percent}% mutation..."
                    )

                    # Read embedding
                    embedding = self.read_embedding_file(emb_file)
                    if embedding is None:
                        print(f"        Failed to read embedding for {file_stem}")
                        continue

                    # Find corresponding mutated sequences
                    mut_seq_file = (
                        seq_type_dir
                        / self.target_model
                        / str(length_percent)
                        / f"{file_stem}.txt"
                    )
                    mutated_sequences = []
                    if mut_seq_file.exists():
                        mutated_sequences = self.read_sequence_file(mut_seq_file)
                        print(
                            f"        Found {len(mutated_sequences)} mutated sequences"
                        )

                    # Find original sequence
                    orig_seq_file_fasta = (
                        orig_seq_type_dir
                        / self.target_model
                        / str(length_percent)
                        / f"{gene_name}.fasta"
                    )
                    orig_seq_file_txt = (
                        orig_seq_type_dir
                        / self.target_model
                        / str(length_percent)
                        / f"{gene_name}.txt"
                    )

                    original_sequences = []
                    if orig_seq_file_fasta.exists():
                        original_sequences = self.read_sequence_file(
                            orig_seq_file_fasta
                        )
                        print(f"        Found original sequence in FASTA")
                    elif orig_seq_file_txt.exists():
                        original_sequences = self.read_sequence_file(orig_seq_file_txt)
                        print(f"        Found original sequence in TXT")

                    original_sequence = (
                        original_sequences[0] if original_sequences else ""
                    )

                    # If we have multiple mutated sequences, process each one
                    if mutated_sequences:
                        for idx, mut_seq in enumerate(mutated_sequences):
                            # Create row data
                            row_data = {
                                "model": self.target_model,
                                "gene_name": gene_name,
                                "sequence_type": seq_type.lower(),
                                "sequence_length_percent": length_percent,
                                "mutation_percentage": mutation_percent,
                                "original_sequence": original_sequence,
                                "mutated_sequence": mut_seq,
                                "sequence": mut_seq,  # Keep the actual mutated sequence
                                "mutated_sequence_index": idx,  # Track which mutated sequence this is
                            }

                            # Add embedding dimensions
                            for i, val in enumerate(embedding):
                                row_data[f"embedding_dim_{i}"] = val

                            self.data.append(row_data)

                        print(
                            f"        Added {len(mutated_sequences)} mutated sequences for {gene_name}"
                        )
                    else:
                        # If no mutated sequences found, create entry with original sequence
                        print(
                            f"        Warning: No mutated sequences found for {file_stem}"
                        )
                        row_data = {
                            "model": self.target_model,
                            "gene_name": gene_name,
                            "sequence_type": seq_type.lower(),
                            "sequence_length_percent": length_percent,
                            "mutation_percentage": mutation_percent,
                            "original_sequence": original_sequence,
                            "mutated_sequence": original_sequence,
                            "sequence": original_sequence,
                            "mutated_sequence_index": 0,
                        }

                        # Add embedding dimensions
                        for i, val in enumerate(embedding):
                            row_data[f"embedding_dim_{i}"] = val

                        self.data.append(row_data)

    def build_dataframe(self):
        """Build the complete DataFrame"""
        print("Building DataFrame...")

        # Process both original and mutated data
        self.process_original_data()
        self.process_mutated_data()

        if not self.data:
            print("No data found!")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(self.data)

        # Sort by model, sequence_type, gene_name, sequence_length_percent, mutation_percentage
        df = df.sort_values(
            [
                "model",
                "sequence_type",
                "gene_name",
                "sequence_length_percent",
                "mutation_percentage",
            ]
        )

        print(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
        print(f"Models: {df['model'].nunique()}")
        print(f"Genes: {df['gene_name'].nunique()}")
        print(f"Sequence types: {df['sequence_type'].unique()}")
        print(f"Length percentages: {sorted(df['sequence_length_percent'].unique())}")
        print(f"Mutation percentages: {sorted(df['mutation_percentage'].unique())}")

        return df

    def save_dataframe(
        self, df, output_path="grover_embedding_dataframe.pkl", format="pickle"
    ):
        """Save DataFrame efficiently"""
        print(f"Saving DataFrame to {output_path}...")

        if format == "pickle":
            df.to_pickle(output_path)
        elif format == "parquet":
            df.to_parquet(output_path)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "hdf5":
            df.to_hdf(output_path, key="embeddings", mode="w")

        print(f"DataFrame saved successfully!")

        # Also save a summary
        summary = {
            "shape": df.shape,
            "model": self.target_model,
            "genes": df["gene_name"].unique().tolist(),
            "sequence_types": df["sequence_type"].unique().tolist(),
            "length_percentages": sorted(df["sequence_length_percent"].unique()),
            "mutation_percentages": sorted(df["mutation_percentage"].unique()),
            "embedding_dimensions": len(
                [col for col in df.columns if col.startswith("embedding_dim_")]
            ),
            "total_rows": len(df),
            "original_sequences": len(df[df["mutation_percentage"] == 0]),
            "mutated_sequences": len(df[df["mutation_percentage"] > 0]),
        }

        with open(output_path.replace(".pkl", "_summary.txt"), "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")


def main():
    """Main function to create the DataFrame for GROVER model only"""
    print("Starting DataFrame creation for GROVER model...")

    # Initialize builder for GROVER model
    builder = EmbeddingDataFrameBuilder(target_model="GROVER")

    # Build DataFrame
    df = builder.build_dataframe()

    if df.empty:
        print("No data found to create DataFrame")
        return

    # Save in multiple formats for efficiency
    builder.save_dataframe(df, "grover_embedding_dataframe.pkl", "pickle")
    builder.save_dataframe(df, "grover_embedding_dataframe.parquet", "parquet")

    # Also save a sample as CSV for inspection
    sample_df = df.head(100)  # Smaller sample for GROVER only
    builder.save_dataframe(sample_df, "grover_embedding_dataframe_sample.csv", "csv")

    print("DataFrame creation completed!")
    print(f"Full dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Show some basic statistics
    print(f"\nDataset Statistics:")
    print(f"- Genes: {df['gene_name'].nunique()}")
    print(f"- Sequence types: {list(df['sequence_type'].unique())}")
    print(f"- Length percentages: {sorted(df['sequence_length_percent'].unique())}")
    print(f"- Mutation percentages: {sorted(df['mutation_percentage'].unique())}")
    print(f"- Original sequences: {len(df[df['mutation_percentage'] == 0])}")
    print(f"- Mutated sequences: {len(df[df['mutation_percentage'] > 0])}")

    return df


if __name__ == "__main__":
    df = main()

import pandas as pd
import pickle


def unpickle_and_save_csv():
    """Load pickle file and save as CSV"""
    try:
        # Load the pickle file
        df = pd.read_pickle("grover_embedding_dataframe.pkl")
        print(f"DataFrame loaded with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Show first few rows
        print(f"\nFirst 3 rows:")
        print(df.head(3))

        # Save as full CSV
        df.to_csv("grover_embedding_dataframe_full.csv", index=False)
        print(f"\nSaved full DataFrame as CSV with {len(df)} rows")

        # Show summary statistics
        print(f"\nDataset Summary:")
        print(f"- Total rows: {len(df)}")
        print(f"- Total columns: {len(df.columns)}")
        print(f'- Genes: {df["gene_name"].nunique()}')
        print(f'- Sequence types: {list(df["sequence_type"].unique())}')
        print(f'- Length percentages: {sorted(df["sequence_length_percent"].unique())}')
        print(f'- Mutation percentages: {sorted(df["mutation_percentage"].unique())}')
        print(f'- Original sequences: {len(df[df["mutation_percentage"] == 0])}')
        print(f'- Mutated sequences: {len(df[df["mutation_percentage"] > 0])}')

        # Show embedding dimensions
        embedding_cols = [col for col in df.columns if col.startswith("embedding_dim_")]
        print(f"- Embedding dimensions: {len(embedding_cols)}")

        return df

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    df = unpickle_and_save_csv()

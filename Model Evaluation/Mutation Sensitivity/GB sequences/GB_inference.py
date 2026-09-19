import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

# Optional dependency required by songlab/gpn-brassicales.
try:
    import gpn.model  # noqa: F401
except Exception:
    pass


device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DNA_BASES = ["A", "T", "C", "G"]

models = [
    ["GROVER", "PoetschLab/GROVER"],
    ["DNABERT-2", "zhihan1996/DNABERT-2-117M"],
    ["GPN", "songlab/gpn-brassicales"],
    ["NT500M", "InstaDeepAI/nucleotide-transformer-500m-human-ref"],
    ["NT2.5B-MS", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"],
    ["NT2.5B-1K", "InstaDeepAI/nucleotide-transformer-2.5b-1000g"],
    ["Hyena-1k", "LongSafari/hyenadna-tiny-1k-seqlen-hf"],
    ["Hyena-16k", "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"],
    ["Hyena-32k", "LongSafari/hyenadna-small-32k-seqlen-hf"],
    ["GenaLM", "AIRI-Institute/gena-lm-bigbird-base-t2t"],
    ["DNABERT-S", "zhihan1996/DNABERT-S"],
    ["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf"],
    ["Hyena-450k", "LongSafari/hyenadna-medium-450k-seqlen-hf"],
    ["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf"],
]

DNABERT_POOLER_MODELS = {"zhihan1996/DNABERT-2-117M", "zhihan1996/DNABERT-S"}

# Server run configuration.
CONFIG = {
    "dataset_name": "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs",
    "output_root": "./mutation_sensitivity_outputs",
    "mutation_percents": list(range(5, 101, 5)),
    "num_mutations_per_percent": 100,
    "batch_size": 64,
    "chunk_size": 256,
    "max_length": 200,
    "run_coding": True,
    "run_noncoding": True,
    "subsample_enabled": True,
    "samples_per_label": 500,
    "subsample_strategy": "first",  # "first" or "random"
    "subsample_seed": 42,
    "sequence_limit": None,
    "save_mutated_sequences": True,
    "save_mutated_embeddings": True,
    "save_original_embeddings": True,
    "resume": True,
}

# Set to all model names if you want full run.
MODELS_TO_RUN = [
    "GROVER",
    "Hyena-1k",
    "Hyena-16k",
    "Hyena-32k",
    "Hyena-160k",
    "GenaLM",
    "NT500M",
    "GPN",
]


def mutater(percent, seq):
    seq_length = len(seq)
    mutation_number = round(percent * seq_length / 100)
    mutation_number = min(mutation_number, seq_length)

    mutation_positions = random.sample(range(seq_length), mutation_number)
    seq_list = list(seq)

    for i in mutation_positions:
        og_base = seq_list[i]
        new_base = random.choice(DNA_BASES)
        while new_base == og_base:
            new_base = random.choice(DNA_BASES)
        seq_list[i] = new_base

    return "".join(seq_list)


def build_mutation_batch(sequences, percent, num_mutations):
    mutated_sequences = []
    parent_indices = []
    mutation_instance = []

    for seq_idx, seq in enumerate(sequences):
        for i in range(num_mutations):
            mutated_sequences.append(mutater(percent, seq))
            parent_indices.append(seq_idx)
            mutation_instance.append(i)

    return mutated_sequences, np.array(parent_indices), np.array(mutation_instance)


def prepare_work_df(config):
    dataset = load_dataset(config["dataset_name"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    full_dataset = concatenate_datasets([train_dataset, test_dataset])

    meta_df = pd.DataFrame(
        {
            "seq": full_dataset["seq"],
            "label": full_dataset["label"],
        }
    )
    meta_df["orig_index"] = np.arange(len(meta_df), dtype=np.int32)

    selected_labels = []
    if config["run_coding"]:
        selected_labels.append(0)
    if config["run_noncoding"]:
        selected_labels.append(1)
    if len(selected_labels) == 0:
        raise ValueError("Both run_coding and run_noncoding are False. Nothing to run.")

    meta_df = meta_df[meta_df["label"].isin(selected_labels)].reset_index(drop=True)

    if config.get("subsample_enabled", False):
        strategy = str(config.get("subsample_strategy", "first")).lower()
        per_label = int(config.get("samples_per_label", 500))
        sampled_parts = []

        for label in selected_labels:
            label_df = meta_df[meta_df["label"] == label]
            take_n = min(per_label, len(label_df))

            if strategy == "first":
                label_sample = label_df.head(take_n)
            elif strategy == "random":
                label_sample = label_df.sample(
                    n=take_n,
                    random_state=config.get("subsample_seed", SEED),
                )
            else:
                raise ValueError("subsample_strategy must be 'first' or 'random'.")

            sampled_parts.append(label_sample)

        meta_df = pd.concat(sampled_parts, ignore_index=True)

        if strategy == "random":
            meta_df = meta_df.sample(
                frac=1.0,
                random_state=config.get("subsample_seed", SEED),
            ).reset_index(drop=True)

    if config["sequence_limit"] is not None:
        meta_df = meta_df.head(min(config["sequence_limit"], len(meta_df))).copy()

    work_df = meta_df[["orig_index", "seq", "label"]].copy().reset_index(drop=True)
    work_df["sequence_type"] = np.where(work_df["label"] == 0, "coding", "non-coding")

    lengths = (
        pd.Series([len(s) for s in work_df["seq"]])
        .value_counts()
        .sort_index()
        .to_dict()
    )
    print(f"Selected sequences: {len(work_df)}")
    print("Label counts (0=coding, 1=non-coding):")
    print(work_df["label"].value_counts().sort_index())
    print(f"Observed lengths: {lengths}")

    return work_df


def load_model_and_tokenizer(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model_name == "zhihan1996/DNABERT-2-117M":
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config,
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    return tokenizer, model


def generate_embeddings_batched(
    model_name, tokenizer, model, sequences, batch_size, max_length
):
    embeddings = []
    total = len(sequences)

    for start_idx in range(0, total, batch_size):
        batch = sequences[start_idx : start_idx + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        if model_name in DNABERT_POOLER_MODELS:
            if (
                isinstance(outputs, (tuple, list))
                and len(outputs) > 1
                and outputs[1] is not None
            ):
                result = outputs[1]
            else:
                result = outputs.last_hidden_state[:, 0, :]
        else:
            result = outputs.last_hidden_state.mean(dim=1)

        embeddings.append(result.detach().cpu().numpy().astype(np.float32))

        done = min(start_idx + batch_size, total)
        if done % (batch_size * 50) == 0 or done == total:
            print(f"    Embedded {done}/{total} sequences")

    return np.concatenate(embeddings, axis=0)


def _to_fixed_size_list_array(matrix):
    rows, dims = matrix.shape
    flat = pa.array(matrix.reshape(rows * dims), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dims)


def compute_progress_totals(config, selected_models, total_sequences):
    chunk_size = config["chunk_size"]
    num_chunks = math.ceil(total_sequences / chunk_size)
    mutation_percents = config["mutation_percents"]

    total_units = len(selected_models) * num_chunks * len(mutation_percents)
    completed_units = 0

    if config.get("resume", False):
        output_root = Path(config["output_root"])
        for model_folder, _ in selected_models:
            model_output_dir = output_root / model_folder
            if not model_output_dir.exists():
                continue
            for chunk_id in range(num_chunks):
                for pct in mutation_percents:
                    out_file = (
                        model_output_dir / f"chunk_{chunk_id:06d}_pct_{pct:03d}.parquet"
                    )
                    if out_file.exists():
                        completed_units += 1

    completed_units = min(completed_units, total_units)
    return total_units, completed_units


def write_results_parquet(
    out_file,
    model_folder,
    global_idx_chunk,
    label_chunk,
    parent_indices,
    mutation_percent,
    mutation_instance,
    mutated_sequences,
    distances,
    cosine_sims,
    mutated_embeddings,
    original_embeddings_for_rows,
    save_mutated_sequences,
    save_mutated_embeddings,
    save_original_embeddings,
):
    labels_for_rows = label_chunk[parent_indices]
    seq_types = np.where(labels_for_rows == 0, "coding", "non-coding")
    row_count = len(distances)

    cols = {
        "model": pa.array([model_folder] * row_count),
        "global_sequence_index": pa.array(
            global_idx_chunk[parent_indices], type=pa.int32()
        ),
        "label": pa.array(labels_for_rows.astype(np.int8), type=pa.int8()),
        "sequence_type": pa.array(seq_types),
        "mutation_percent": pa.array(
            np.full(row_count, mutation_percent, dtype=np.int16), type=pa.int16()
        ),
        "mutation_instance": pa.array(
            mutation_instance.astype(np.int16), type=pa.int16()
        ),
        "distance": pa.array(distances.astype(np.float32), type=pa.float32()),
        "cosine_similarity": pa.array(
            cosine_sims.astype(np.float32), type=pa.float32()
        ),
    }

    if save_mutated_sequences:
        cols["mutated_sequence"] = pa.array(mutated_sequences)

    if save_mutated_embeddings:
        cols["mutated_embedding"] = _to_fixed_size_list_array(
            mutated_embeddings.astype(np.float32)
        )

    if save_original_embeddings:
        cols["original_embedding"] = _to_fixed_size_list_array(
            original_embeddings_for_rows.astype(np.float32)
        )

    table = pa.table(cols)
    pq.write_table(table, out_file, compression="zstd")


def run_model(model_folder, model_name, work_df, config, overall_pbar=None):
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    model_output_dir = output_root / model_folder
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(model_name)

    chunk_size = config["chunk_size"]
    num_chunks = math.ceil(len(work_df) / chunk_size)

    print(f"Now starting with: {model_folder}")
    print(f"Total sequences: {len(work_df)} | Chunks: {num_chunks}")
    if overall_pbar is not None:
        overall_pbar.set_description(f"Overall | {model_folder}")

    model_start = time.time()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, len(work_df))
        chunk_df = work_df.iloc[start_idx:end_idx]

        seq_chunk = chunk_df["seq"].tolist()
        label_chunk = chunk_df["label"].to_numpy(dtype=np.int8)
        global_idx_chunk = chunk_df["orig_index"].to_numpy(dtype=np.int32)

        expected_files = [
            model_output_dir / f"chunk_{chunk_id:06d}_pct_{pct:03d}.parquet"
            for pct in config["mutation_percents"]
        ]
        if config["resume"] and all(p.exists() for p in expected_files):
            print(f"  Skipping chunk {chunk_id + 1}/{num_chunks} (already complete)")
            continue

        print(f"  Processing chunk {chunk_id + 1}/{num_chunks} (size={len(seq_chunk)})")
        original_embeddings = generate_embeddings_batched(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            sequences=seq_chunk,
            batch_size=config["batch_size"],
            max_length=config["max_length"],
        )
        original_norms = np.linalg.norm(original_embeddings, axis=1)

        for pct in config["mutation_percents"]:
            out_file = model_output_dir / f"chunk_{chunk_id:06d}_pct_{pct:03d}.parquet"
            if config["resume"] and out_file.exists():
                continue

            mutated_sequences, parent_indices, mutation_instance = build_mutation_batch(
                seq_chunk,
                percent=pct,
                num_mutations=config["num_mutations_per_percent"],
            )

            mutated_embeddings = generate_embeddings_batched(
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                sequences=mutated_sequences,
                batch_size=config["batch_size"],
                max_length=config["max_length"],
            )

            original_embeddings_for_rows = original_embeddings[parent_indices]
            distances = np.linalg.norm(
                mutated_embeddings - original_embeddings_for_rows,
                axis=1,
            ).astype(np.float32)

            mutated_norms = np.linalg.norm(mutated_embeddings, axis=1)
            denom = mutated_norms * original_norms[parent_indices]
            dots = np.sum(mutated_embeddings * original_embeddings_for_rows, axis=1)
            cosine_sims = np.divide(
                dots,
                denom,
                out=np.zeros_like(dots, dtype=np.float32),
                where=denom > 0,
            ).astype(np.float32)

            write_results_parquet(
                out_file=out_file,
                model_folder=model_folder,
                global_idx_chunk=global_idx_chunk,
                label_chunk=label_chunk,
                parent_indices=parent_indices,
                mutation_percent=pct,
                mutation_instance=mutation_instance,
                mutated_sequences=mutated_sequences,
                distances=distances,
                cosine_sims=cosine_sims,
                mutated_embeddings=mutated_embeddings,
                original_embeddings_for_rows=original_embeddings_for_rows,
                save_mutated_sequences=config.get("save_mutated_sequences", True),
                save_mutated_embeddings=config.get("save_mutated_embeddings", True),
                save_original_embeddings=config.get("save_original_embeddings", False),
            )
            if overall_pbar is not None:
                overall_pbar.update(1)

            del (
                mutated_sequences,
                parent_indices,
                mutation_instance,
                mutated_embeddings,
                original_embeddings_for_rows,
                distances,
                mutated_norms,
                denom,
                dots,
                cosine_sims,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"    Saved: {out_file.name}")

        del original_embeddings, original_norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_end = time.time()
    print(
        f"Done with model {model_folder} in {round(model_end - model_start, 2)} seconds"
    )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    start = time.time()

    output_root = Path(CONFIG["output_root"]).resolve()
    CONFIG["output_root"] = str(output_root)
    print(f"Device: {device}")
    print(f"Saving outputs to: {CONFIG['output_root']}")

    work_df = prepare_work_df(CONFIG)

    available_map = {name: hf_id for name, hf_id in models}
    missing = [name for name in MODELS_TO_RUN if name not in available_map]
    if missing:
        raise ValueError(f"Unknown model aliases in MODELS_TO_RUN: {missing}")

    selected = [[name, available_map[name]] for name in MODELS_TO_RUN]
    print(f"Models to run: {[name for name, _ in selected]}")

    total_units, completed_units = compute_progress_totals(
        CONFIG, selected, len(work_df)
    )
    print(
        f"Overall progress targets: {completed_units}/{total_units} chunk-percent tasks already complete"
    )

    overall_pbar = tqdm(
        total=total_units,
        initial=completed_units,
        desc="Overall progress",
        unit="task",
        dynamic_ncols=True,
    )
    try:
        for model_folder, model_name in selected:
            run_model(
                model_folder, model_name, work_df, CONFIG, overall_pbar=overall_pbar
            )
    finally:
        overall_pbar.close()

    end = time.time()
    print(f"All requested models completed in {round(end - start, 2)} seconds")


if __name__ == "__main__":
    main()

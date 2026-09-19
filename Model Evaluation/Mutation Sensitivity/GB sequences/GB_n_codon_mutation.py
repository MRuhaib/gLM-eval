"""
GB_codon_inference.py
---------------------
Codon-degeneracy mutation sensitivity experiment for DNA language models.

Hypothesis:
    Coding sequences are buffered against synonymous mutations — especially at
    the wobble (3rd) codon position — because the genetic code is degenerate.
    A DNA LLM that has internalised codon structure should show smaller embedding
    shifts for synonymous mutations than for missense mutations in coding sequences,
    while non-coding (intergenomic) sequences should show no such differentiation.

Experiment design:
    For every sequence × ORF (frame 0/1/2) × codon position (1st/2nd/3rd) ×
    consequence track (synonymous / missense) × n_mutations × mutation_instance:

        1. Pre-screen all 66 complete codons in the ORF for eligibility:
           a codon is eligible for a track if it has at least one alternative base
           at the target codon position that produces the required consequence type.
           Nonsense (stop-creating) mutations are excluded from both tracks.

        2. Randomly sample n codons (without replacement) from the eligible pool,
           mutate the target codon position in each to a randomly chosen alternative
           of the required consequence type.  Every site in the resulting sequence
           differs from the original at exactly that one position per selected codon.

        3. Embed the mutated sequence with each DNA LLM.

        4. Record L2 distance and cosine similarity to the original embedding,
           together with full provenance metadata for downstream analysis.

    Output is one Parquet file per (model, chunk), with one row per mutated sequence.

Key analysis axes (done separately, not in this script):
    - Dose-response: embedding distance vs. n_mutations, stratified by
      consequence_class, codon_position, and sequence_type.
    - Within position-2 (wobble): synonymous vs missense curves in coding vs
      non-coding — the primary codon-degeneracy signature.
    - ORF quality: orf_stop_count is the number of in-frame stop codons; the ORF
      with fewest stop codons is the best candidate for the true reading frame and
      should show the strongest signal in coding sequences.
    - Transition vs transversion: a secondary biochemical axis.
    - Model comparison: which DNA LLMs show the strongest codon-degeneracy signal.
"""

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


# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DNA_BASES = ["A", "T", "C", "G"]

# Standard genetic code (DNA, uppercase).  '*' = stop codon.
GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Transitions: purine <-> purine (A<->G) or pyrimidine <-> pyrimidine (C<->T).
# All other base changes are transversions.
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}

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

CONFIG = {
    # --- Dataset ---
    "dataset_name": "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs",
    "output_root": "./codon_mutation_outputs",
    # --- Mutation design ---
    # n_values: number of codons mutated simultaneously in one mutated sequence.
    # Fibonacci-ish spacing gives denser sampling at low n (where per-mutation
    # effects are clearest) and sparser at high n (diminishing marginal returns).
    "n_values": [1, 2, 3, 5, 8, 13, 20],
    # k_samples: independent random draws per (seq, orf, codon_pos, consequence, n).
    # Each draw picks a different random subset of n eligible codons.
    "k_samples": 25,
    # ORFs to evaluate: 0 = frame starting at seq[0], 1 = seq[1], 2 = seq[2].
    # All three yield 66 complete codons for 200-nt sequences.
    "orfs": [0, 1, 2],
    # Codon positions within each codon: 0 = 1st nucleotide, 1 = 2nd, 2 = 3rd (wobble).
    "codon_positions": [0, 1, 2],
    # Consequence tracks.  'synonymous' eligible codons are rare at positions 0/1
    # (the script skips a track if no eligible codons exist), so in practice the
    # synonymous track is active primarily at codon position 2.
    "consequence_classes": ["synonymous", "missense"],
    # --- Embedding ---
    "batch_size": 64,
    # Smaller chunk_size than original because mutation count per sequence is ~10x higher.
    # chunk_size=128 keeps the per-chunk mutated-sequence buffer at ~250 k sequences,
    # which is manageable in RAM (~750 MB for 768-dim float32 embeddings).
    "chunk_size": 128,
    "max_length": 200,
    # --- Sequence selection ---
    "run_coding": True,
    "run_noncoding": True,
    "subsample_enabled": True,
    "samples_per_label": 500,
    "subsample_strategy": "first",  # "first" | "random"
    "subsample_seed": 42,
    "sequence_limit": None,
    # --- Output ---
    "save_mutated_sequences": True,  # store the full mutated sequence string
    "save_mutated_embeddings": False,  # large; enable only if needed for deep dives
    "save_original_embeddings": False,
    "resume": True,
}

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


# ---------------------------------------------------------------------------
# Codon utilities
# ---------------------------------------------------------------------------


def get_codon_frame(seq: str, frame: int) -> list[tuple[str, int]]:
    """
    Extract all complete (length-3) codons from `seq` in the given reading frame.

    For 200-nt sequences all three frames yield exactly 66 complete codons:
        frame 0: codons at [0:3], [3:6], ..., [195:198]   (2 nt remainder)
        frame 1: codons at [1:4], [4:7], ..., [196:199]   (1 nt remainder)
        frame 2: codons at [2:5], [5:8], ..., [197:200]   (0 nt remainder)

    Returns:
        List of (codon_str, global_start_position) tuples.
    """
    codons = []
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if len(codon) == 3:
            codons.append((codon, i))
    return codons


def get_alternatives(
    codon: str, codon_pos: int, consequence: str
) -> list[tuple[str, str]]:
    """
    Return all (new_base, new_codon) pairs that:
        - change position `codon_pos` (0/1/2) in `codon`
        - produce the requested `consequence` ('synonymous' or 'missense')
        - do NOT create a stop codon (nonsense mutations are excluded from both tracks)

    'synonymous': new amino acid == original amino acid
    'missense'  : new amino acid != original amino acid, original is not a stop

    Rationale for excluding nonsense from both tracks:
        Stop-creating mutations are biologically catastrophic and would dominate
        the embedding shift signal, obscuring the codon-degeneracy comparison
        between synonymous and conservative missense mutations.
        They can be analysed separately if desired using the consequence counts
        stored per row (n_nonsense).
    """
    orig_base = codon[codon_pos]
    orig_aa = GENETIC_CODE.get(codon, "?")
    results = []

    for new_base in DNA_BASES:
        if new_base == orig_base:
            continue
        new_codon = codon[:codon_pos] + new_base + codon[codon_pos + 1 :]
        new_aa = GENETIC_CODE.get(new_codon, "?")

        if new_aa == "*":
            continue  # skip stop-creating mutations

        if consequence == "synonymous" and new_aa == orig_aa:
            results.append((new_base, new_codon))
        elif consequence == "missense" and new_aa != orig_aa and orig_aa != "*":
            results.append((new_base, new_codon))

    return results


def classify_single_point_mutation(
    orig_codon: str, codon_pos: int, new_base: str
) -> tuple[str, bool]:
    """
    Classify a single point mutation within a codon.

    Returns:
        mutation_type : 'synonymous' | 'missense' | 'nonsense' | 'stop_lost'
        is_transition : True if the base change is a transition (A<->G or C<->T),
                        False if it is a transversion.
    """
    orig_base = orig_codon[codon_pos]
    new_codon = orig_codon[:codon_pos] + new_base + orig_codon[codon_pos + 1 :]
    orig_aa = GENETIC_CODE.get(orig_codon, "?")
    new_aa = GENETIC_CODE.get(new_codon, "?")

    if orig_aa == "*":
        mut_type = "stop_lost" if new_aa != "*" else "synonymous"
    elif new_aa == "*":
        mut_type = "nonsense"
    elif orig_aa == new_aa:
        mut_type = "synonymous"
    else:
        mut_type = "missense"

    is_transition = (orig_base, new_base) in TRANSITIONS
    return mut_type, is_transition


def count_stop_codons(seq: str, frame: int) -> int:
    """
    Count in-frame stop codons for the given reading frame.

    Used as an ORF quality metric: the frame with the fewest stop codons is the
    strongest candidate for the actual coding reading frame.  Stored as
    `orf_stop_count` in the output for downstream stratification.
    """
    return sum(
        1 for codon, _ in get_codon_frame(seq, frame) if GENETIC_CODE.get(codon) == "*"
    )


# ---------------------------------------------------------------------------
# Core mutation batch builder
# ---------------------------------------------------------------------------


def build_codon_mutation_batch(
    sequences: list[str], config: dict
) -> tuple[list[str], list[dict], np.ndarray]:
    """
    Generate all codon-structured mutated sequences for a list of sequences.

    For every combination of:
        seq_idx  × orf  × codon_pos  × consequence  × n  × k (mutation_instance)

    the function:
        1. Computes the pool of eligible codons for (orf, codon_pos, consequence).
        2. If len(eligible) < n, skips this (n, consequence) pair — cannot sample
           n distinct codons without replacement.
        3. Otherwise, samples n codons without replacement, and for each mutates
           `codon_pos` to a randomly chosen alternative of the required consequence.
        4. Records the mutated sequence and full provenance metadata.

    The n_synonymous / n_missense / n_nonsense counts on each row are verification
    fields — for well-formed controlled tracks they should equal n (or 0 for the
    other types).  Edge cases (e.g., a codon whose only synonymous alternative
    happens to create a stop via some indirect mechanism) are caught here.

    Args:
        sequences : list of raw DNA sequence strings for the current chunk
        config    : CONFIG dict (reads n_values, k_samples, orfs,
                    codon_positions, consequence_classes)

    Returns:
        mutated_seqs   : flat list of mutated sequence strings
        metadata_rows  : list of dicts, one per mutated sequence, with keys:
                            orf, orf_stop_count, codon_position, n_mutations,
                            consequence_class, mutation_instance,
                            n_synonymous, n_missense, n_nonsense, n_eligible_codons
        parent_indices : int32 array mapping each mutated sequence back to its
                         original sequence index within `sequences`
    """
    n_values = config["n_values"]
    k_samples = config["k_samples"]
    orfs = config["orfs"]
    codon_positions = config["codon_positions"]
    consequence_classes = config["consequence_classes"]

    mutated_seqs = []
    metadata_rows = []
    parent_indices = []

    for seq_idx, seq in enumerate(sequences):

        # Precompute ORF quality metric for each requested frame.
        stop_counts = {f: count_stop_codons(seq, f) for f in orfs}

        for orf in orfs:
            codons = get_codon_frame(seq, orf)
            if not codons:
                continue

            for codon_pos in codon_positions:
                for consequence in consequence_classes:

                    # ----------------------------------------------------------
                    # Build the eligible codon pool for this (orf, codon_pos,
                    # consequence) combination once; reuse across all (n, k).
                    # ----------------------------------------------------------
                    eligible = []
                    for codon_idx, (codon, global_start) in enumerate(codons):
                        alts = get_alternatives(codon, codon_pos, consequence)
                        if alts:
                            eligible.append((codon_idx, codon, global_start, alts))

                    # If no codon supports this consequence at this position,
                    # skip entirely.  This is the expected behaviour for
                    # 'synonymous' at codon positions 0 and 1 (very few codons
                    # have synonymous alternatives there).
                    if not eligible:
                        continue

                    n_eligible = len(eligible)

                    for n in n_values:
                        if n_eligible < n:
                            # Cannot draw n distinct codons; skip.
                            continue

                        for k in range(k_samples):
                            # Sample n codons without replacement from eligible pool.
                            selected = random.sample(eligible, n)

                            # Apply all n point mutations to a fresh copy of the seq.
                            seq_list = list(seq)
                            n_syn = n_mis = n_non = 0

                            for _, codon, global_start, alts in selected:
                                new_base, _ = random.choice(alts)
                                global_pos = global_start + codon_pos
                                seq_list[global_pos] = new_base

                                # Classify the individual point mutation for
                                # verification and secondary analysis.
                                mut_type, _ = classify_single_point_mutation(
                                    codon, codon_pos, new_base
                                )
                                if mut_type == "synonymous":
                                    n_syn += 1
                                elif mut_type == "missense":
                                    n_mis += 1
                                elif mut_type == "nonsense":
                                    n_non += 1
                                # stop_lost counted in neither (very rare)

                            mutated_seqs.append("".join(seq_list))
                            parent_indices.append(seq_idx)
                            metadata_rows.append(
                                {
                                    "orf": orf,
                                    "orf_stop_count": stop_counts[orf],
                                    "codon_position": codon_pos,
                                    "n_mutations": n,
                                    "consequence_class": consequence,
                                    "mutation_instance": k,
                                    "n_synonymous": n_syn,
                                    "n_missense": n_mis,
                                    "n_nonsense": n_non,
                                    "n_eligible_codons": n_eligible,
                                }
                            )

    return mutated_seqs, metadata_rows, np.array(parent_indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# Dataset preparation  (unchanged from original)
# ---------------------------------------------------------------------------


def prepare_work_df(config: dict) -> pd.DataFrame:
    dataset = load_dataset(config["dataset_name"])
    full_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

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
    if not selected_labels:
        raise ValueError("Both run_coding and run_noncoding are False.")

    meta_df = meta_df[meta_df["label"].isin(selected_labels)].reset_index(drop=True)

    if config.get("subsample_enabled", False):
        strategy = str(config.get("subsample_strategy", "first")).lower()
        per_label = int(config.get("samples_per_label", 500))
        parts = []

        for label in selected_labels:
            label_df = meta_df[meta_df["label"] == label]
            take_n = min(per_label, len(label_df))

            if strategy == "first":
                parts.append(label_df.head(take_n))
            elif strategy == "random":
                parts.append(
                    label_df.sample(
                        n=take_n, random_state=config.get("subsample_seed", SEED)
                    )
                )
            else:
                raise ValueError("subsample_strategy must be 'first' or 'random'.")

        meta_df = pd.concat(parts, ignore_index=True)
        if strategy == "random":
            meta_df = meta_df.sample(
                frac=1.0, random_state=config.get("subsample_seed", SEED)
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


# ---------------------------------------------------------------------------
# Model loading & embedding  (unchanged from original)
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if model_name == "zhihan1996/DNABERT-2-117M":
        cfg = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, config=cfg
        ).to(device)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    model.eval()
    return tokenizer, model


def generate_embeddings_batched(
    model_name: str,
    tokenizer,
    model,
    sequences: list[str],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    if len(sequences) == 0:
        return np.empty((0, 0), dtype=np.float32)

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


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def _to_fixed_size_list_array(matrix: np.ndarray) -> pa.FixedSizeListArray:
    rows, dims = matrix.shape
    flat = pa.array(matrix.reshape(rows * dims), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dims)


def write_results_parquet(
    out_file: Path,
    model_folder: str,
    global_idx_chunk: np.ndarray,
    label_chunk: np.ndarray,
    parent_indices: np.ndarray,
    metadata_rows: list[dict],
    mutated_sequences: list[str],
    distances: np.ndarray,
    cosine_sims: np.ndarray,
    mutated_embeddings: np.ndarray,
    original_embeddings_for_rows: np.ndarray,
    save_mutated_sequences: bool,
    save_mutated_embeddings: bool,
    save_original_embeddings: bool,
) -> None:
    """
    Write one Parquet file for a single (model, chunk) pair.

    Schema
    ------
    Provenance:
        model                 str      model alias (e.g. 'GROVER')
        global_sequence_index int32    index in the full concatenated dataset
        label                 int8     0 = coding, 1 = non-coding
        sequence_type         str      'coding' | 'non-coding'

    ORF & codon structure:
        orf                   int8     reading frame (0, 1, or 2)
        orf_stop_count        int8     number of in-frame stop codons (ORF quality)
        codon_position        int8     position within codon (0=1st, 1=2nd, 2=3rd/wobble)
        n_mutations           int8     number of codons simultaneously mutated
        consequence_class     str      intended consequence track ('synonymous'|'missense')
        mutation_instance     int16    random sample index (0 .. k_samples-1)

    Consequence verification:
        n_synonymous          int8     actual number of synonymous mutations applied
        n_missense            int8     actual number of missense mutations applied
        n_nonsense            int8     actual number of nonsense mutations applied
        n_eligible_codons     int8     pool size available for this (orf,pos,consequence)

    Embedding distances:
        distance              float32  L2 (Euclidean) distance between embeddings
        cosine_similarity     float32  cosine similarity between embeddings

    Optional:
        mutated_sequence      str      full 200-nt mutated sequence
        mutated_embedding     list<f32> embedding of mutated sequence
        original_embedding    list<f32> embedding of original sequence
    """
    labels_for_rows = label_chunk[parent_indices]
    seq_types = np.where(labels_for_rows == 0, "coding", "non-coding")
    row_count = len(distances)

    # Unpack metadata lists into typed arrays.
    def _meta(key, dtype):
        return np.array([m[key] for m in metadata_rows], dtype=dtype)

    cols = {
        # --- Provenance ---
        "model": pa.array([model_folder] * row_count),
        "global_sequence_index": pa.array(
            global_idx_chunk[parent_indices], type=pa.int32()
        ),
        "label": pa.array(labels_for_rows.astype(np.int8), type=pa.int8()),
        "sequence_type": pa.array(seq_types),
        # --- ORF & codon structure ---
        "orf": pa.array(_meta("orf", np.int8), type=pa.int8()),
        "orf_stop_count": pa.array(_meta("orf_stop_count", np.int8), type=pa.int8()),
        "codon_position": pa.array(_meta("codon_position", np.int8), type=pa.int8()),
        "n_mutations": pa.array(_meta("n_mutations", np.int8), type=pa.int8()),
        "consequence_class": pa.array([m["consequence_class"] for m in metadata_rows]),
        "mutation_instance": pa.array(
            _meta("mutation_instance", np.int16), type=pa.int16()
        ),
        # --- Consequence verification ---
        "n_synonymous": pa.array(_meta("n_synonymous", np.int8), type=pa.int8()),
        "n_missense": pa.array(_meta("n_missense", np.int8), type=pa.int8()),
        "n_nonsense": pa.array(_meta("n_nonsense", np.int8), type=pa.int8()),
        "n_eligible_codons": pa.array(
            _meta("n_eligible_codons", np.int8), type=pa.int8()
        ),
        # --- Distances ---
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

    pq.write_table(pa.table(cols), out_file, compression="zstd")


# ---------------------------------------------------------------------------
# Per-model main loop
# ---------------------------------------------------------------------------


def run_model(
    model_folder: str,
    model_name: str,
    work_df: pd.DataFrame,
    config: dict,
    overall_pbar=None,
) -> None:
    output_root = Path(config["output_root"])
    model_output_dir = output_root / model_folder
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(model_name)

    chunk_size = config["chunk_size"]
    num_chunks = math.ceil(len(work_df) / chunk_size)

    print(f"\nStarting model: {model_folder}")
    print(f"Total sequences: {len(work_df)} | Chunks: {num_chunks}")
    if overall_pbar is not None:
        overall_pbar.set_description(f"Overall | {model_folder}")

    model_start = time.time()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(work_df))
        chunk_df = work_df.iloc[start_idx:end_idx]

        seq_chunk = chunk_df["seq"].tolist()
        label_chunk = chunk_df["label"].to_numpy(dtype=np.int8)
        global_idx_chunk = chunk_df["orig_index"].to_numpy(dtype=np.int32)

        out_file = model_output_dir / f"chunk_{chunk_id:06d}.parquet"

        if config["resume"] and out_file.exists():
            print(f"  Skipping chunk {chunk_id + 1}/{num_chunks} (already complete)")
            if overall_pbar is not None:
                overall_pbar.update(1)
            continue

        print(f"  Chunk {chunk_id + 1}/{num_chunks} (size={len(seq_chunk)})")

        # --- Embed originals ---
        print("    Embedding original sequences...")
        original_embeddings = generate_embeddings_batched(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            sequences=seq_chunk,
            batch_size=config["batch_size"],
            max_length=config["max_length"],
        )
        original_norms = np.linalg.norm(original_embeddings, axis=1)

        # --- Generate all codon mutations for this chunk ---
        print("    Generating codon mutation batch...")
        mutated_seqs, metadata_rows, parent_indices = build_codon_mutation_batch(
            seq_chunk, config
        )
        print(
            f"    {len(mutated_seqs):,} mutated sequences generated "
            f"({len(mutated_seqs) / len(seq_chunk):.0f} per original)"
        )

        if len(mutated_seqs) == 0:
            print("    No eligible codon mutations found for this chunk; skipping.")
            if overall_pbar is not None:
                overall_pbar.update(1)
            del original_embeddings, original_norms, mutated_seqs, metadata_rows, parent_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # --- Embed mutations ---
        print("    Embedding mutated sequences...")
        mutated_embeddings = generate_embeddings_batched(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            sequences=mutated_seqs,
            batch_size=config["batch_size"],
            max_length=config["max_length"],
        )

        # --- Compute distances ---
        original_embeddings_for_rows = original_embeddings[parent_indices]

        distances = np.linalg.norm(
            mutated_embeddings - original_embeddings_for_rows, axis=1
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

        # --- Write ---
        write_results_parquet(
            out_file=out_file,
            model_folder=model_folder,
            global_idx_chunk=global_idx_chunk,
            label_chunk=label_chunk,
            parent_indices=parent_indices,
            metadata_rows=metadata_rows,
            mutated_sequences=mutated_seqs,
            distances=distances,
            cosine_sims=cosine_sims,
            mutated_embeddings=mutated_embeddings,
            original_embeddings_for_rows=original_embeddings_for_rows,
            save_mutated_sequences=config.get("save_mutated_sequences", True),
            save_mutated_embeddings=config.get("save_mutated_embeddings", False),
            save_original_embeddings=config.get("save_original_embeddings", False),
        )
        print(f"    Saved {out_file.name} ({len(mutated_seqs):,} rows)")

        if overall_pbar is not None:
            overall_pbar.update(1)

        # --- Memory cleanup ---
        del (
            mutated_seqs,
            metadata_rows,
            parent_indices,
            mutated_embeddings,
            original_embeddings_for_rows,
            distances,
            mutated_norms,
            denom,
            dots,
            cosine_sims,
        )
        del original_embeddings, original_norms
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done: {model_folder} in {round(time.time() - model_start, 2)}s")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def compute_progress_totals(
    config: dict, selected_models: list, total_sequences: int
) -> tuple[int, int]:
    """
    One output file per (model, chunk).  Total units = n_models × n_chunks.
    """
    num_chunks = math.ceil(total_sequences / config["chunk_size"])
    total_units = len(selected_models) * num_chunks
    completed = 0

    if config.get("resume", False):
        output_root = Path(config["output_root"])
        for model_folder, _ in selected_models:
            model_dir = output_root / model_folder
            if not model_dir.exists():
                continue
            for chunk_id in range(num_chunks):
                if (model_dir / f"chunk_{chunk_id:06d}.parquet").exists():
                    completed += 1

    return total_units, min(completed, total_units)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    output_root = Path(CONFIG["output_root"]).resolve()
    CONFIG["output_root"] = str(output_root)

    print(f"Device      : {device}")
    print(f"Output root : {output_root}")

    work_df = prepare_work_df(CONFIG)

    available_map = {name: hf_id for name, hf_id in models}
    missing = [n for n in MODELS_TO_RUN if n not in available_map]
    if missing:
        raise ValueError(f"Unknown model aliases: {missing}")

    selected = [[name, available_map[name]] for name in MODELS_TO_RUN]
    print(f"Models to run: {[n for n, _ in selected]}")

    # --- Quick sanity: estimate mutations per sequence ---
    _sample_seq = work_df["seq"].iloc[0]
    _sample_muts, _, _ = build_codon_mutation_batch([_sample_seq], CONFIG)
    print(f"Estimated mutations per sequence: {len(_sample_muts):,}")
    print(
        f"Estimated total embedding calls per model: "
        f"{len(_sample_muts) * len(work_df):,}"
    )

    total_units, completed_units = compute_progress_totals(
        CONFIG, selected, len(work_df)
    )
    print(f"Progress: {completed_units}/{total_units} chunk tasks already done")

    overall_pbar = tqdm(
        total=total_units,
        initial=completed_units,
        desc="Overall",
        unit="chunk",
        dynamic_ncols=True,
    )
    try:
        for model_folder, model_name in selected:
            run_model(
                model_folder, model_name, work_df, CONFIG, overall_pbar=overall_pbar
            )
    finally:
        overall_pbar.close()

    print(f"\nAll models completed in {round(time.time() - start, 2)}s")


if __name__ == "__main__":
    main()

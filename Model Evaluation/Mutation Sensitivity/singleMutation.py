import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig  # for DNABERT-2
import gpn.model  # For the GPN model to work properly
from more_itertools import batched

device = "cuda" if torch.cuda.is_available() else "cpu"

# Models which have higher distances for non-coding sequences:
"""
models = [
    ["GROVER", "PoetschLab/GROVER"],
    # ["DNABERT-2", "zhihan1996/DNABERT-2-117M"], #Weird, irregular variation in distances
    ["GPN", "songlab/gpn-brassicales"],  # Higher for coding
    [
        "NT500M",
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    ],  # Higher for coding in .75x sequences?
    [
        "Hyena-1k",
        "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    ],  # Higher for coding in .75x and 1x sequences.
]
    # Do batch inference for above models.
"""
models = [
    [
        "Hyena-16k",
        "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    ],  # Higher for non-coding in 1x sequences.
    [
        "Hyena-32k",
        "LongSafari/hyenadna-small-32k-seqlen-hf",
    ],  # Higher for coding in .25x; non-coding is more only by a small margin in the other 3.
    ["DNABERT-S", "zhihan1996/DNABERT-S"],  # Overlapping in .5x and 1x sequences.
    # ["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf"], #Mostly overlapping distances
    [
        "Hyena-450k",
        "LongSafari/hyenadna-medium-450k-seqlen-hf",
    ],  # Non-coding distances are higher by a small margin.
    ["GenaLM", "AIRI-Institute/gena-lm-bigbird-base-t2t"],
    # ^^ Gives some error for the .74 and 1x sequences as well; some tensor mismatch.
    ["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf"],
    # ^^ 750k and 1m sequences i.e, .75 and 1 don't run; needs more ram.
    [
        "NT2.5B-MS",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    ],  # Higher for non-coding in .75x sequences??
    [
        "NT2.5B-1K",
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    ],  # Distance separation between non/coding increases with sequence length, interestingly
]


types = ["Coding", "Non-coding"]
fractions = ["25", "50", "75", "100"]


def generateEmbeddings(modelName, batch):
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    # Just for DNABERT-2
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained(
        modelName,
        trust_remote_code=True,
        config=config if modelName == "zhihan1996/DNABERT-2-117M" else None,
    ).to(device)
    embeddings = []
    sequences = []

    isOgSeq = isinstance(batch, str)
    if isOgSeq:
        sequences = [batch]
    else:
        for mutatedSeq in batch:
            sequences.append(mutatedSeq[0])
            embeddings.append(mutatedSeq[1:])

    # Use batching to speed up inference for smaller models:
    inputs = tokenizer(
        sequences,  # Contains the 4 mutated sequences; one for each base (including the original)
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = (
        outputs[1].cpu().squeeze().numpy()
        if modelName == "zhihan1996/DNABERT-2-117M"
        or modelName == "zhihan1996/DNABERT-S"
        else outputs.last_hidden_state.cpu().mean(dim=1).squeeze().numpy()
    )

    if isOgSeq:
        embedding = [item for item in result]
        embeddings.append(embedding)
    else:
        if result.ndim == 1:
            embedding = [item for item in result]
            embeddings.append([embedding, newBases[0]])
        else:
            for i, element in enumerate(
                result
            ):  # since the embeddings are batched here
                embedding = [item for item in element]
                # Have to keep track of which embedding corresponds to which of the 4 NTs is substituted.
                embeddings[i].append(embedding)

    if isOgSeq:
        embeddings = embeddings[0]
    return embeddings


def main(seqClass="Original"):
    start = time.time()
    """
    Dataframe structure: One for each model; no. of mutated sequences for each fraction = len(fraction); if euclidean and cosine distance = 0 --> it's the orignal sequence
    sequence type (coding/noncoding) - model - context window fraction (25/50/75/100) - new base (A/T/C/G) - euclidean distance - cosine distance
    """
    columns = [
        "model",
        "sequence_type",  # Coding/Non-coding
        "fraction",  # 25/50/75/100
        "seqLen",
        "og_base",  # A/T/C/G
        "position",  # i ∈ (0, len(seq))
        "new_base",  # A/T/C/G
        "mut_type",  # Transition/Transversion
        "euclidean",
        "cosine",
    ]
    data = []
    for model in models:
        torch.cuda.empty_cache()
        modelFolder = model[0]
        for seqType in types:
            print(
                "Now starting with model:", modelFolder, "And sequence type:", seqType
            )
            for fraction in fractions:
                if (model == "Hyena-1m" or model == "GenaLM") and fraction in [
                    "75",
                    "100",
                ]:
                    continue  # Since these give errors.

                mainPath = f"{seqType}/{modelFolder}/{fraction}"
                seqDirectory = (
                    f"drive/MyDrive/inference/{seqClass} Sequences/{mainPath}"
                    if modelFolder[:2] != "NT"
                    else f"drive/MyDrive/inference/{seqClass} Sequences/{seqType}/NT/{fraction}"
                )

                for filename in Path(seqDirectory).glob("*.txt"):
                    name = str(filename).split("/")[-1]
                    ogSequence = ""

                    with open(filename, "r+") as f:
                        ogSequence = str(f.read())

                    seqLen = len(ogSequence)
                    ogEmbedding = generateEmbeddings(model[1], ogSequence)
                    pointMutatedSeqs = []

                    for i in range(len(ogSequence)):
                        # each i corresponds to one nucleotide that has been mutated to the other 3 possible ones.
                        bases = ["A", "T", "C", "G"]
                        purines = ["A", "G"]
                        pyrimidines = ["T", "C"]
                        for newBase in bases:
                            mutSeq = list(ogSequence)
                            ogBase = mutSeq[i]
                            if mutSeq[i] != newBase:
                                mutSeq[i] = newBase
                                if (ogBase in purines and newBase in purines) or (
                                    ogBase in pyrimidines and newBase in pyrimidines
                                ):
                                    mutType = "Transition"
                                else:
                                    mutType = "Transversion"
                                pointMutatedSeqs.append(
                                    ["".join(mutSeq), ogBase, i, newBase, mutType]
                                )
                    specificBatches = {
                        "Hyena-16k": 25,
                        "Hyena-32k": 25,
                        "GenaLM": 25,
                        "Hyena-450k": 20,
                        "Hyena-1m": 10,
                        "DNABERT-S": 10,
                    }
                    batchSize = (
                        100
                        if modelFolder
                        not in [
                            "DNABERT-S",
                            "Hyena-450k",
                            "GenaLM",
                            "Hyena-1m",
                            "Hyena-16k",
                            "Hyena-32k",
                        ]
                        else specificBatches[modelFolder]
                    )

                    batchedInputs = list(batched(pointMutatedSeqs, batchSize))

                    for batch in batchedInputs:
                        pointMutatedEmbeddings = generateEmbeddings(model[1], batch)
                        for element in pointMutatedEmbeddings:
                            (ogBase, position, newBase, mutType, embedding) = tuple(
                                element
                            )
                            cosineDist = cosine(ogEmbedding, embedding)
                            euclideanDist = euclidean(ogEmbedding, embedding)
                            data.append(
                                [
                                    modelFolder,
                                    seqType,
                                    fraction,
                                    seqLen,
                                    ogBase,
                                    position,
                                    newBase,
                                    mutType,
                                    euclideanDist,
                                    cosineDist,
                                ]
                            )

                end = time.time()
                print(
                    f"Done with {fraction}% of Maxlen sequences for {modelFolder} in {round(end - start, 2)} seconds."
                )
        print(f"All done with {modelFolder}.")
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f"allDistances.csv")


if __name__ == "__main__":
    main()

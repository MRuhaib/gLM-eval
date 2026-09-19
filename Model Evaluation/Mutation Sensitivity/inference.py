import torch
import numpy as np
import time
import json
from Bio import SeqIO
from pathlib import Path
import os
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig  # for DNABERT-2
import gpn.model  # For the GPN model to work properly

device = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: Hyena-1m and Gena-LM can't generate embeddings for the .75x and 1x context-window length sequences.

models = [
    ["GROVER", "PoetschLab/GROVER"],
    ["DNABERT-2", "zhihan1996/DNABERT-2-117M"],
    ["GPN", "songlab/gpn-brassicales"],
    ["NT500M", "InstaDeepAI/nucleotide-transformer-500m-human-ref"],
    ["NT2.5B-MS", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"],
    ["NT2.5B-1K", "InstaDeepAI/nucleotide-transformer-2.5b-1000g"],
    ["Hyena-1k", "LongSafari/hyenadna-tiny-1k-seqlen-hf"],
    # Do batch inference for above models.
    ["Hyena-16k", "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"],
    ["Hyena-32k", "LongSafari/hyenadna-small-32k-seqlen-hf"],
    ["GenaLM", "AIRI-Institute/gena-lm-bigbird-base-t2t"],
    # Gives some error for the .74 and 1x sequences as well; some tensor mismatch.
    ["DNABERT-S", "zhihan1996/DNABERT-S"],
    ["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf"],
    ["Hyena-450k", "LongSafari/hyenadna-medium-450k-seqlen-hf"],
    ["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf"],
    # 750k and 1m sequences i.e, .75 and 1 don't run; needs more ram.
]

types = ["Coding", "Non-coding", "Mixed"]
numMutations = 100  # Number of mutations for a specific value of x
fractions = ["25", "50", "75", "100"]


def generateEmbeddings(modelName, sequences):
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    config = BertConfig.from_pretrained(
        "zhihan1996/DNABERT-2-117M"
    )  # Just for DNABERT-2
    model = AutoModel.from_pretrained(
        modelName,
        trust_remote_code=True,
        config=config if modelName == "zhihan1996/DNABERT-2-117M" else None,
    ).to(device)
    embeddings = []

    # For large models, avoid batching:
    for sequence in sequences:
        inputs = tokenizer(
            sequence,
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
        embedding = [item for item in result]
        embeddings.append(embedding)

    """
    #Use batching to speed up inference for smaller models:
    inputs = tokenizer(
        sequences, #Remember, this contains the 100 randomly mutated sequences for that particular sequence length!
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = outputs[1].cpu().squeeze().numpy() if modelName == "zhihan1996/DNABERT-2-117M" or modelName == "zhihan1996/DNABERT-S" else outputs.last_hidden_state.cpu().mean(dim=1).squeeze().numpy()

    for element in result: #since the embeddings are batched here
        embedding = [item for item in element]
        embeddings.append(embedding)
    """

    return embeddings


def main(seqType, seqClass):
    start = time.time()
    for model in models:
        modelFolder = model[0]
        print("Now starting with:", modelFolder)
        for fraction in fractions:
            if (model == "Hyena-1m" or model == "GenaLM") and fraction in ["75", "100"]:
                continue  # Since these give errors.

            mainPath = f"{seqType}/{modelFolder}/{fraction}"
            seqDirectory = (
                f"./Sequences/{seqClass} Sequences/{mainPath}"
                if modelFolder[:2] != "NT"
                else f"./Sequences/{seqClass} Sequences/{seqType}/NT/{fraction}"
            )
            newDirectory = f"./Embeddings/{seqClass}/{mainPath}"

            os.makedirs(newDirectory, exist_ok=True)
            for filename in Path(seqDirectory).glob("*.txt"):
                name = str(filename).split("/")[-1]
                print("Now doing,", filename)
                sequences = []

                with open(filename, "r+") as f:
                    sequences = eval(f.read())
                    # Since all the 100 mutations of a specific x value are stored in the form of a list in their respective text files.

                if isinstance(sequences, str):
                    sequences = [sequences]
                    # Since the orignal sequences are just single strings stored in the text files.

                embeddings = generateEmbeddings(model[1], sequences)

                # Store in dataframes instead!!
                with open(f"{newDirectory}/{name}", "w+") as f:
                    f.write(str(embeddings))

            end = time.time()
            print(
                f"Done with {fraction}% of Maxlen sequences for {modelFolder} in {round(end - start, 2)} seconds."
            )


if __name__ == "__main__":
    seqType = types[1]
    seqClass = "Original"
    main(seqType, seqClass)

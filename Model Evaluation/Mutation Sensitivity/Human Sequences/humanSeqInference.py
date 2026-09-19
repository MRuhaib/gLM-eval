import torch
import pandas as pd
import time
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig  # for DNABERT-2
import gpn.model  # For the GPN model to work properly
from more_itertools import batched

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Save EVERYTHING into a single dataframe (write after each of the 5000 sequences for each model), with the following features:
"model", "sequence_type", "fraction", "seq_len", "seq_no", "mut_percentage", "old_A_count", "old_T_count", "old_C_count", "old_G_count", "new_A_count", "new_T_count", "new_C_count", "new_G_count", "Transitions", "Transversions", "Cosine", "Euclidean"
string   coding/non-cod  25/50/75/100  modelLen   1-1000   5, 10, ... , 50        int            int            int            int            int            int            int            int            int             int         float      float      
"""
models = [
    #["GPN", "songlab/gpn-brassicales"],
    ["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf"],
    ["Hyena-450k", "LongSafari/hyenadna-medium-450k-seqlen-hf"],
    ["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf"],
]
smolModels = [
    #["GROVER", "PoetschLab/GROVER"],
    #["Hyena-1k", "LongSafari/hyenadna-tiny-1k-seqlen-hf"],
    #["GPN", "songlab/gpn-brassicales"],
    #["DNABERT-2", "zhihan1996/DNABERT-2-117M"],
    ["NT500M", "InstaDeepAI/nucleotide-transformer-500m-human-ref"],
    ["Hyena-16k", "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"],
    ["NT2.5B-MS", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"],
    ["NT2.5B-1K", "InstaDeepAI/nucleotide-transformer-2.5b-1000g"],
    # Do batch inference for above models.
]

largeModels = [
    #["Hyena-32k", "LongSafari/hyenadna-small-32k-seqlen-hf"], have to do for non-coding!
    # Gives some error for the .74 and 1x sequences as well; some tensor mismatch.
    #["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf"],
    #["Hyena-450k", "LongSafari/hyenadna-medium-450k-seqlen-hf"],
    #["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf"],
    #["GenaLM", "AIRI-Institute/gena-lm-bigbird-base-t2t"],
    #["NT2.5B-MS", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"],
    #["NT2.5B-1K", "InstaDeepAI/nucleotide-transformer-2.5b-1000g"],
    ["DNABERT-S", "zhihan1996/DNABERT-S"],
    ["DNABERT-2", "zhihan1996/DNABERT-2-117M"],
    # 750k and 1m sequences i.e, .75 and 1 don't run; needs more ram.
]

geneTypes = ["Coding", "Non-coding"]
seqTypes = ["CDS", "Genes"]

def seqReader(file):
    sequences = []
    for sequence in SeqIO.parse(file, "fasta"):
        seq = str(sequence.seq)
        seqID = str(sequence.id)
        if len(seq) > 100000: #ultra-long outlier sequences; they ruin inference runs.
            continue
        #19805 coding genes and 18757 non-coding genes from the ncbi human genes dataset
        sequences.append([seq, seqID])
    return sequences

def generateSingleEmbedding(modelName, seq):
    #torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    # Just for DNABERT-2
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained(
        modelName,
        trust_remote_code=True,
        config=config if modelName == "zhihan1996/DNABERT-2-117M" else None,
    ).to(device)
    inputs = tokenizer(
        seq,
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

    return embedding

def generateBatchedEmbeddings(modelName, batch):
    #torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
    # Just for DNABERT-2
    if modelName == "zhihan1996/DNABERT-2-117M":
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    model = AutoModel.from_pretrained(
        modelName,
        trust_remote_code=True,
        config=config if modelName == "zhihan1996/DNABERT-2-117M" else None,
    ).to(device)
    embeddings = []
    sequences = []

    for seq in batch:
        sequences.append(seq[0])
        embeddings.append(seq[1:])

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

    if result.ndim == 1:
        embedding = [item for item in result]
        embeddings.append([embedding])
    else:
        for i, element in enumerate(
            result
        ):  # since the embeddings are batched here
            embedding = [item for item in element]
            # Have to keep track of which embedding corresponds to which of the 4 NTs is substituted.
            embeddings[i].append(embedding)

    return embeddings



def main(seqClass="Original"):
    start = time.time()

    for model in models:
        data = []
        modelFolder = model[0]

        for seqType in geneTypes:
            print("Now starting with model:", modelFolder, "and sequence type:", seqType)

            #sequences = seqReader(f"{seqType}/filtered_{seqType}.fasta") 
            #Remember, this has a 1000 sequences with lengths between 100 and 5000!

            sequences = seqReader(f"{seqType}/data/gene.fna") #All genes!
            print(f"There are a total of {len(sequences)} sequences of human {seqType} type.")

            for n, element in enumerate(sequences):
                sequence = element[0]
                seqID = element[1]
                seqLen = len(sequence)
                sequences[n] = [sequence, seqID, seqType, seqLen]
                #ogEmbedding = generateSingleEmbedding(model[1], sequence)

            
            for i, seq in enumerate(sequences):
                embedding = generateSingleEmbedding(model[1], seq[0])
                currentTime = time.time()
                if (i+1) % 100 == 0 or (i+1) % 100 == len(sequences) % 100:
                    print(f"Done with {i+1} sequences in {round(currentTime - start, 2)} seconds..")
                data.append(seq[1:] + [embedding])
                
            """
            batchSize = 5
            batchedInputs = list(batched(sequences, batchSize))

            finishedCount = 0
            for i, batch in enumerate(batchedInputs):
                finishedCount += len(batch)
                allEmbeddings = generateBatchedEmbeddings(model[1], batch)
                currentTime = time.time()
                if finishedCount % 100 == 0 or finishedCount % 100 == len(sequences) % 100:
                    print(f"Done with {finishedCount} sequences in {round(currentTime - start, 2)} seconds..")
                    torch.cuda.empty_cache()
                
                for element in allEmbeddings:
                    #(seqID, seqType, seqLen, embedding) = tuple(element)
                    data.append(element)
            """
            

            end = time.time()
            print(f"All done with {modelFolder} - {seqType} sequences in {round(end-start, 2)} seconds.")
            embeddingDf = pd.DataFrame(data, columns=["sequence_ID", "sequence_type", "sequence_length", "embedding"])
            embeddingDf.to_csv(f"Embeddings/Coding-NonCoding/{modelFolder}_allEmbeddings.csv")


if __name__ == "__main__":
    main()

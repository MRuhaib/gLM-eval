import torch
import pandas as pd
import time
import random
from Bio import SeqIO
from scipy.spatial.distance import cosine, euclidean
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig  # for DNABERT-2
import gpn.model  # For the GPN model to work properly
from more_itertools import batched
from transformers import logging
import warnings

warnings.simplefilter('ignore')
logging.set_verbosity_warning()
#maybe these can get rid of those annoying warning logs?

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Save EVERYTHING into a single dataframe (write after each of the 5000 sequences for each model), with the following features:
"model", "sequence_type", "fraction", "seq_len", "seq_no", "mut_percentage", "old_A_count", "old_T_count", "old_C_count", "old_G_count", "new_A_count", "new_T_count", "new_C_count", "new_G_count", "Transitions", "Transversions", "Cosine", "Euclidean"
string   coding/non-cod  25/50/75/100  modelLen   1-1000   5, 10, ... , 50        int            int            int            int            int            int            int            int            int             int         float      float      
"""

smolModels = [
    ["GROVER", "PoetschLab/GROVER"],
    ["GPN", "songlab/gpn-brassicales"],
    ["Hyena-1k", "LongSafari/hyenadna-tiny-1k-seqlen-hf"],
    #["DNABERT-2", "zhihan1996/DNABERT-2-117M"],
    ["NT500M", "InstaDeepAI/nucleotide-transformer-500m-human-ref"],
    ["Hyena-16k", "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"],
    ["NT2.5B-MS", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"],
    ["NT2.5B-1K", "InstaDeepAI/nucleotide-transformer-2.5b-1000g"],
    # Do batch inference for above models.
]

largeModels = [
    ["Hyena-16k", "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"],
    ["Hyena-32k", "LongSafari/hyenadna-small-32k-seqlen-hf"],
    #["GenaLM", "AIRI-Institute/gena-lm-bigbird-base-t2t"],
    # Gives some error for the .74 and 1x sequences as well; some tensor mismatch.
    ["DNABERT-S", "zhihan1996/DNABERT-S"],
    #["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf"],
    #["Hyena-450k", "LongSafari/hyenadna-medium-450k-seqlen-hf"],
    #["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf"],
    # 750k and 1m sequences i.e, .75 and 1 don't run; needs more ram.
]

geneTypes = ["Coding", "Non-coding"]
seqTypes = ["CDS", "Genes"]

def seqReader(file):
    sequences = []
    for sequence in SeqIO.parse(file, "fasta"):
        seq = str(sequence.seq)
        seqID = str(sequence.id)
        sequences.append([seq, seqID])
    return sequences

def mutater(x, seq):
    seqLength = len(seq)
    mutationNumber = round(x * seqLength / 100)

    # This generate unique mutation positions, avoids checking each time.
    mutationPositions = random.sample(range(seqLength), mutationNumber)

    seqList = list(seq)

    for i in mutationPositions:
        ogBase = seqList[i]
        newBase = random.choice(["A", "T", "C", "G"])
        while newBase == ogBase:  # Ensure the new base is different
            newBase = random.choice(["A", "T", "C", "G"])
        seqList[i] = newBase

    # Convert back to string
    return "".join(seqList)

def getMutTypeNumber(og, mut):
    purines = ["A", "G"]
    pyrimidines = ["T", "C"]
    transitions, transversions = 0, 0
    for o, m in zip(og, mut):
        if o != m:  # Mutation detected
            if (o in purines and m in purines) or (o in pyrimidines and m in pyrimidines):
                transitions += 1
            else:
                transversions += 1
    return transitions, transversions

def generateSingleEmbedding(modelName, seq):
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
    columns = ["model", "sequence_type", "seq_len", "gene_ID", "mut_percentage", "old_A_count", "old_T_count", "old_C_count", "old_G_count", "new_A_count", "new_T_count", "new_C_count", "new_G_count", "Transitions", "Transversions", "Cosine", "Euclidean"]

    for model in smolModels:
        #torch.cuda.empty_cache()
        data = []
        #embeddings = []
        modelFolder = model[0]

        for seqType in seqTypes:
            print("Now starting with model:", modelFolder, "and sequence type:", seqType)

            sequences = seqReader(f"{seqType}/filtered_{seqType}.fasta") 
            #Remember, this has a 1000 sequences with lengths between 100 and 5000!

            for n, element in enumerate(sequences):
                currentTime = time.time()
                sequence = element[0]
                seqID = element[1]
                seqLen = len(sequence)

                if n != 0 and n % 100 == 0:
                    print(f"Done with {n} sequences in {round(currentTime - start, 2)} seconds..")

                ogEmbedding = generateSingleEmbedding(model[1], sequence)

                #embeddings.append([seqID, seqType, ogEmbedding])

                old_A = sequence.count('A')
                old_T = sequence.count('T')
                old_C = sequence.count('C')
                old_G = sequence.count('G')

                for x in range(5, 51, 5):
                    mutatedSequences = [] #Ayyooo u r resetting this each time, dats y u r only getting x = 50 daa!!

                    for i in range(100):
                        mutSeq = mutater(x, sequence)
                        new_A = mutSeq.count('A')
                        new_T = mutSeq.count('T')
                        new_C = mutSeq.count('C')
                        new_G = mutSeq.count('G')
                        transitions, transversions = getMutTypeNumber(sequence, mutSeq)
                        mutatedSequences.append([mutSeq, new_A, new_T, new_C, new_G, transitions, transversions])

                batchSize = 200 if modelFolder in ["GPN", "GROVER", "Hyena-1k"] else 100
                batchedInputs = list(batched(mutatedSequences, batchSize))

                for batch in batchedInputs:
                    pointMutatedEmbeddings = generateBatchedEmbeddings(model[1], batch)
                    for element in pointMutatedEmbeddings:
                        (new_A, new_T, new_C, new_G, transitions, transversions, embedding) = tuple(element)
                        cosineDist = cosine(ogEmbedding, embedding)
                        euclideanDist = euclidean(ogEmbedding, embedding)
                        data.append([modelFolder, seqType, seqLen, seqID, x, old_A, old_T, old_C, old_G, new_A, new_T, new_C, new_G, transitions, transversions, euclideanDist, cosineDist])

        end = time.time()
        print(f"All done with {modelFolder} in {round(end-start, 2)} seconds.")
        df = pd.DataFrame(data, columns=columns)
        #embeddingDf = pd.DataFrame(embeddings, columns=["sequence_ID", "sequence_type", "embedding"])

        df.to_csv(f"Distances/CDS-Genes/{modelFolder}_allDistances_50.csv", index=False)
        #embeddingDf.to_csv(f"Embeddings/CDS-Genes/{modelFolder}_allEmbeddings.csv", index=False)


if __name__ == "__main__":
    main()

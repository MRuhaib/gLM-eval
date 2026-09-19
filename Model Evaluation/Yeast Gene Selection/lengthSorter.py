import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import time

modeLengths = {
    "DNABERT2": 500,
    "GPN": 500,
    "GROVER": 500,
    "NT": 1000,
    "DNABERTS": 10000,
    "GenaLM": 36000,
    "Hyena": {
        "1k": 1000,
        "16k": 16000,
        "32k": 32000,
        "160k": 160000,
        "450k": 450000,
        "1m": 1000000,
    },
}


def seqLenFinder(file):
    seqLen = 0
    for sequence in SeqIO.parse(file, "fasta"):
        seqLen = len(sequence.seq)
        break
    return seqLen


allGenes = []
allLengths = []
max = 0
min = 20000

parseTimeStart = time.time()
for filename in Path("All Genes").glob("*.fasta"):
    gene = str(filename).split("\\")[1].rstrip(".fasta")
    allGenes.append([gene, seqLenFinder(filename)])
    allLengths.append(seqLenFinder(filename))
    if seqLenFinder(filename) > max:
        max = seqLenFinder(filename)
    if seqLenFinder(filename) < min:
        min = seqLenFinder(filename)
parseTimeEnd = time.time()

print(
    "Done parsing all fasta files in",
    round(parseTimeEnd - parseTimeStart, 2),
    "; Max length:",
    max,
    "; Min length:",
    min,
)

sortTimeStart = time.time()

allLengths.sort()
finalRanking = []

for length in allLengths:
    for gene in allGenes:
        if gene[1] == length and gene not in finalRanking:
            finalRanking.append(gene)

sortTimeEnd = time.time()

print(
    f"Done parsing {len(finalRanking)} fasta files in",
    round(sortTimeEnd - sortTimeStart, 2),
)
with open("sortedGenes.txt", "w+") as f:
    f.write(str(finalRanking))

"""
Now, sort according to their lengths, 
then classify according to 25, 50 and 75 percent of the models' context windows. 
Select the genes with the most appropriate lengths
"""

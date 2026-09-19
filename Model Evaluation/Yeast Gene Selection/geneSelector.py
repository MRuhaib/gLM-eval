import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import time

with open("sortedGenes.txt", "r+") as f:
    sortedGenes = eval(f.read())

print(len(sortedGenes))

hundredGenes = {i: [] for i in range(50, 5001, 50)}
seqLen = 50

"""
for gene in sortedGenes:
    geneName, geneLen = gene
    if 50 < geneLen < 5500:
        geneDist = abs(geneLen - seqLen)
        print(geneDist)
        while geneDist <= 20:
            hundredGenes[seqLen].append([geneName, geneLen, geneDist])
        else:
            seqLen += 50
            if seqLen > 5000:
                break
"""

geneIndex = 0

for seqLen in range(50, 5001, 50):
    # Find genes within 20 bases of the target length
    while geneIndex < len(sortedGenes):
        geneName, geneLen = sortedGenes[geneIndex]
        geneDist = abs(geneLen - seqLen)

        # If gene length is too small, move to next gene
        if geneLen < seqLen - 20:
            geneIndex += 1
            continue

        # If gene length is too large, break and move to next target
        if geneLen > seqLen + 20:
            break

        # Gene is within range, add it
        hundredGenes[seqLen].append([geneName, geneLen, geneDist])
        geneIndex += 1

finalGenes = {}

for key, values in hundredGenes.items():
    if not values:  # No genes found for this length
        finalGenes[key] = [None, None, 20]
        continue

    minDist = float("inf")
    minGene, minLen = None, None
    print(f"Length {key}: {len(values)} candidates")
    for gene in values:
        if gene[2] < minDist:
            minDist = gene[2]
            minLen = gene[1]
            minGene = gene[0]
    finalGenes[key] = [minGene, minLen, minDist]

with open("hundredGenes.txt", "w+") as f:
    f.write(str(hundredGenes))

with open("finalGenes.txt", "w+") as f:
    f.write(str(finalGenes))

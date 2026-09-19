from Bio import SeqIO
from pathlib import Path


def seqReader(file):
    seqLen = 0
    for sequence in SeqIO.parse(file, "fasta"):
        seqLen = len(sequence.seq)
        sequence = sequence.seq
        break
    if seqLen != 0:
        return sequence, seqLen
    else:
        return "", 0


allSeqs = ""
totalLength = 0
count = 0

for filename in Path("All Genes").glob("*.fasta"):
    sequence, seqLen = seqReader(filename)
    allSeqs += sequence
    totalLength += seqLen
    count += 1
    print(totalLength, count)


with open("codingGenesCombined.txt", "w+") as f:
    f.write(str(allSeqs))

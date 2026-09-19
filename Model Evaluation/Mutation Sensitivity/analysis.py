import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean

# Make box plots to quanitfy difference between magnitudes of distances between coding and non coding genes
# Store in a dataframe - one col for coding sequence, other for non coding, then for distance wrt og sequence, and another for params of it.
"""
Take a NT sequence - say AT, then mutate one to the other 3 possible NTs while keeping all the other NTs in the sequence the same, and analyse the difference
"""


models = [
    ["GROVER", "PoetschLab/GROVER", (1, 15)],
    ["DNABERT-2", "zhihan1996/DNABERT-2-117M", (2.5, 7.5)],
    ["GPN", "songlab/gpn-brassicales", (5, 75)],
    ["NT500M", "InstaDeepAI/nucleotide-transformer-500m-human-ref", (2, 12)],
    ["NT2.5B-MS", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species", (0, 20)],
    ["NT2.5B-1K", "InstaDeepAI/nucleotide-transformer-2.5b-1000g", (0, 5)],
    ["Hyena-1k", "LongSafari/hyenadna-tiny-1k-seqlen-hf", (0, 2.5)],
    ["Hyena-16k", "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf", (0, 2.5)],
    ["Hyena-32k", "LongSafari/hyenadna-small-32k-seqlen-hf", (0, 2.5)],
    ["GenaLM", "AIRI-Institute/gena-lm-bigbird-base-t2t", (0, 5)],
    ["DNABERT-S", "zhihan1996/DNABERT-S", (0, 5)],
    ["Hyena-160k", "LongSafari/hyenadna-medium-160k-seqlen-hf", (0, 2.5)],
    ["Hyena-450k", "LongSafari/hyenadna-medium-450k-seqlen-hf", (0, 2.5)],
    ["Hyena-1m", "LongSafari/hyenadna-large-1m-seqlen-hf", (0, 2.5)],
]

types = ["Coding", "Non-coding", "Mixed"]
numMutations = 100  # Number of mutations for a specific value of x
fractions = ["25", "50", "75", "100"]


def mutationEval(seqType):
    # Compare average cosine/euclidean distance between each of the 100 mutated sequence’s embedding with the original embedding, and plot a histogram of the scores - do this for each of the mutation percentages
    start = time.time()
    for model in models:
        modelFolder = model[0]
        print("Now starting with:", modelFolder)
        for fraction in fractions:
            if (model == "Hyena-1m" or model == "GenaLM") and fraction in ["75", "100"]:
                continue  # No embeddings for these 2; so skip

            mainPath = f"{seqType}/{modelFolder}/{fraction}"
            originalEmbeddingsPath = f"./Embeddings/Original/{mainPath}"
            mutatedEmbeddingsPath = f"./Embeddings/Mutated/{mainPath}"
            distancesPath = f"./Results/Distances/{mainPath}"
            os.makedirs(distancesPath, exist_ok=True)

            ogEmbedding, mutatedEmbeddings = [], []
            distancesDict = {}

            for filename in Path(originalEmbeddingsPath).glob("*.txt"):
                with open(filename, "r+") as ogFile:
                    ogEmbedding = (eval(ogFile.read()))[0]
                    # Single embedding, but stored as a 2D array mistakenly.

            for filename in Path(mutatedEmbeddingsPath).glob("*.txt"):
                name = str(filename).split("/")[-1]
                x = name.split("_")[-1].rstrip(".txt")

                distances = {"cosine": [], "euclidean": []}

                with open(filename, "r+") as f:
                    mutatedEmbeddings = eval(f.read())
                # Has 100 embeddings in this; need to find the distances between them and the original one.
                # Store in json files for each model's fraction; {(x) : {cosine: (array of 100 distances), euclidean: (array of 100 distances)

                for embedding in mutatedEmbeddings:
                    cosineDist = cosine(ogEmbedding, embedding)
                    distances["cosine"].append(cosineDist)
                    euclideanDist = euclidean(ogEmbedding, embedding)
                    distances["euclidean"].append(euclideanDist)

                distancesDict[x] = distances

            with open(f"{distancesPath}/distances.json", "w+") as distancesFile:
                json.dump(distancesDict, distancesFile)

            fractionFinish = time.time()
            print(
                f"Done with fraction {fraction}'s embeddings in {round(fractionFinish - start, 2)} seconds."
            )


def xwisePlotter(distancesDict, model, plotPath, fraction):
    """
    1. Mutation percentage(x)-wise:
    Compare average cosine/euclidean distance between each of the 100 mutated sequence’s embedding with the original embedding, and plot a histogram of the scores - do this for each of the mutation percentages
    """

    plt.rcParams["font.size"] = 6
    plt.rcParams["axes.labelsize"] = 6
    plt.rcParams["xtick.labelsize"] = plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["savefig.pad_inches"] = 0.2

    euclideanFig, euclideanAxs = plt.subplots(2, 5, figsize=(8, 8))
    cosineFig, cosineAxs = plt.subplots(2, 5, figsize=(8, 8))

    for mutPercent, distances in distancesDict.items():
        count = int(mutPercent) // 5 - 1
        # to make sure that x's plots are in ascending order
        euclideanDists = distances["euclidean"]
        euclideanAxs[count // 5][count % 5].hist(euclideanDists, bins=20, color="red")
        euclideanAxs[count // 5][count % 5].set_title(f"{mutPercent}% Mutatation")
        euclideanAxs[count // 5][count % 5].set(
            xlim=model[2],  # different limits for different models!
            ylabel="Number of Mutated Embeddings",
            xlabel="Euclidean distance",
        )

        cosineDists = distances["cosine"]
        cosineAxs[count // 5][count % 5].hist(cosineDists, bins=20, color="blue")
        cosineAxs[count // 5][count % 5].set_title(f"{mutPercent}% Mutatation")
        cosineAxs[count // 5][count % 5].set(
            xlim=(-0.2, 2.2),
            ylabel="Number of Mutated Embeddings",
            xlabel="Cosine distance",
        )

    euclideanFig.tight_layout(rect=[0, 0, 1, 0.93])
    euclideanFig.suptitle(
        f"Variation in Euclidean Distances between mutated & original embeddings, for mutation percentages ranging between 5 - 50%. \nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
        fontsize=10,
        y=0.98,
    )

    cosineFig.tight_layout(rect=[0, 0, 1, 0.93])
    cosineFig.suptitle(
        f"Variation in Cosine Distances between mutated & original embeddings, for mutation percentages ranging between 5 - 50%. \nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
        fontsize=10,
        y=0.98,
    )

    # plt.show()
    euclideanFig.savefig(f"{plotPath}/Euclidean.png", dpi=500, bbox_inches="tight")
    cosineFig.savefig(f"{plotPath}/Cosine.png", dpi=500, bbox_inches="tight")


def fractionwisePlotter(distancesDict, model, plotPath, fraction):
    """
    2. Context window/sequence length percentage-wise:
    Average all the distances for a specific mutation; track change in average over x, for that context window. - line plot, one for each fraction, with both cosine and euclidean variation in it
    """

    mutations = [x for x in range(5, 51, 5)]
    euclideanDistsAll = [0] * len(mutations)
    cosineDistsAll = [0] * len(mutations)

    for mutPercent, distances in distancesDict.items():
        index = int(mutPercent) // 5 - 1

        euclideanDists = distances["euclidean"]
        euclideanDistsAll[index] = np.average(euclideanDists)

        cosineDists = distances["cosine"]
        cosineDistsAll[index] = np.average(cosineDists)

    fig, euclideanAxs = plt.subplots()
    euclideanAxs.set_xlabel("Mutation Percentage")
    euclideanAxs.set_ylabel("Averaged Euclidean Distance")
    euclideanAxs.yaxis.label.set_color("red")
    euclideanAxs.plot(mutations, euclideanDistsAll, color="red")
    euclideanAxs.tick_params(axis="y", labelcolor="red")

    cosineAxs = euclideanAxs.twinx()
    # So that both these distances can be plotted together, with their own scales on the y axes and the same x axis i.e., mutation percentage.
    # Adapted from: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html

    cosineAxs.set_ylabel("Averaged Cosine Distance")
    cosineAxs.yaxis.label.set_color("blue")
    cosineAxs.plot(mutations, cosineDistsAll, color="blue")
    cosineAxs.tick_params(axis="y", labelcolor="blue")

    plt.xticks(mutations)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(
        f"Variation in averaged Cosine & Euclidean Distances \nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
        fontsize=10,
        y=0.98,
    )
    # plt.show()
    fig.savefig(f"{plotPath}/Averaged.png", dpi=500, bbox_inches="tight")


def modelwisePlotter(model):
    """
    3. Model-wise:
    compare it over the 4 types of input sequences for a model - single line plot (one each for euclidean and cosine) with 4 lines, one for each fraction
    """
    mutations = [x for x in range(5, 51, 5)]
    euclideanDistsAll = [0] * len(mutations)
    cosineDistsAll = [0] * len(mutations)

    euclideanFig, euclideanAxs = plt.subplots()
    euclideanAxs.set_xlabel("Mutation Percentage")
    euclideanAxs.set_ylabel("Averaged Euclidean Distance")
    euclideanAxs.plot(mutations, codingEuclideanDistsAll, color="green")
    euclideanAxs.plot(mutations, noncodingEuclideanDistsAll, color="orange")
    euclideanAxs.legend(["Coding Sequences", "Non-coding Sequences"])

    cosineFig, cosineAxs = plt.subplots()
    cosineAxs.set_xlabel("Mutation Percentage")
    cosineAxs.set_ylabel("Averaged Cosine Distance")
    cosineAxs.plot(mutations, codingCosineDistsAll, color="green")
    cosineAxs.plot(mutations, noncodingCosineDistsAll, color="orange")
    cosineAxs.legend(["Coding Sequences", "Non-coding Sequences"])

    for fraction in fractions:
        if (model == "Hyena-1m" or model == "GenaLM") and fraction in ["75", "100"]:
            continue  # No embeddings for these 2; so skip

        mainPath = f"{seqType}/{modelFolder}/{fraction}"
        distancesPath = f"./Results/Distances/{mainPath}"

        distancesDict = {}
        with open(f"{distancesPath}/distances.json", "r+") as f:
            distancesDict = json.load(f)

        plotPath = f"./Results/Plots/{mainPath}"
        os.makedirs(plotPath, exist_ok=True)

    plt.xticks(mutations)

    euclideanFig.tight_layout(rect=[0, 0, 1, 0.93])
    euclideanFig.suptitle(
        f"Variation in averaged Euclidean Distances for the two sequence types\nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
        fontsize=10,
        y=0.98,
    )

    cosineFig.tight_layout(rect=[0, 0, 1, 0.93])
    cosineFig.suptitle(
        f"Variation in averaged Cosine Distances for the two sequence types\nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
        fontsize=10,
        y=0.98,
    )

    # plt.show()
    euclideanFig.savefig(f"{plotPath}/Euclidean.png", dpi=500, bbox_inches="tight")
    cosineFig.savefig(f"{plotPath}/Cosine.png", dpi=500, bbox_inches="tight")


def overallPlotter(distancesDict, model, mainPath):
    """
    4. Overall:
    Perform some aggregation on all the above data specific to each model, and then do a comparative analysis of the models.
    """
    pass


def plotter(seqType, choice):

    start = time.time()
    for model in models:
        modelFolder = model[0]
        print("Now starting with:", modelFolder)
        if choice == "model-wise":
            xwisePlotter(model)
        for fraction in fractions:
            if (model == "Hyena-1m" or model == "GenaLM") and fraction in ["75", "100"]:
                continue  # No embeddings for these 2; so skip

            mainPath = f"{seqType}/{modelFolder}/{fraction}"
            distancesPath = f"./Results/Distances/{mainPath}"

            distancesDict = {}
            with open(f"{distancesPath}/distances.json", "r+") as f:
                distancesDict = json.load(f)

            plotPath = f"./Results/Plots/{mainPath}"
            os.makedirs(plotPath, exist_ok=True)

            if choice == "x-wise":
                xwisePlotter(distancesDict, model, plotPath, fraction)
            elif choice == "fraction-wise":
                fractionwisePlotter(distancesDict, model, plotPath, fraction)


def versusPlotter():
    """
    Also, make box plots to quantify the difference between the distances of coding and non coding sequences.
    """
    for model in models:
        modelFolder = model[0]
        print("Now starting with:", modelFolder)
        for fraction in fractions:
            if (model == "Hyena-1m" or model == "GenaLM") and fraction in ["75", "100"]:
                continue  # No embeddings for these 2; so skip

            mainPath = f"{modelFolder}/{fraction}"
            codingPath = f"Coding/{mainPath}"
            noncodingPath = f"Non-coding/{mainPath}"
            distancesPath = f"./Results/Distances/{mainPath}"

            codingDistancesDict, noncodingDistancesDict = {}, {}
            with open(f"./Results/Distances/{codingPath}/distances.json", "r+") as f:
                codingDistancesDict = json.load(f)
            with open(f"./Results/Distances/{noncodingPath}/distances.json", "r+") as f:
                noncodingDistancesDict = json.load(f)

            plotPath = f"./Results/Plots/Comparison/{mainPath}"
            os.makedirs(plotPath, exist_ok=True)

            mutations = [x for x in range(5, 51, 5)]
            codingEuclideanDistsAll = [0] * len(mutations)
            codingCosineDistsAll = [0] * len(mutations)
            noncodingEuclideanDistsAll = [0] * len(mutations)
            noncodingCosineDistsAll = [0] * len(mutations)

            for mutPercent, distances in codingDistancesDict.items():
                index = int(mutPercent) // 5 - 1

                euclideanDists = distances["euclidean"]
                # codingEuclideanDistsAll[index] = np.average(euclideanDists) #For chart plots
                codingEuclideanDistsAll[index] = euclideanDists

                cosineDists = distances["cosine"]
                # codingCosineDistsAll[index] = np.average(cosineDists)
                codingCosineDistsAll[index] = cosineDists

            for mutPercent, distances in noncodingDistancesDict.items():
                index = int(mutPercent) // 5 - 1

                euclideanDists = distances["euclidean"]
                # noncodingEuclideanDistsAll[index] = np.average(euclideanDists) #For chart plots
                noncodingEuclideanDistsAll[index] = euclideanDists

                cosineDists = distances["cosine"]
                # noncodingCosineDistsAll[index] = np.average(cosineDists)
                noncodingCosineDistsAll[index] = cosineDists

            # For the boxplots:
            # Shifting to storing stuff in dataframes since it's easier.
            columns = ["Mutation Percentage", "Distance", "Sequence Type"]
            cosineData, euclideanData = [], []

            for i, percentage in enumerate(mutations):
                for dist in codingCosineDistsAll[i]:
                    cosineData.append([percentage, dist, "Coding"])
                for dist in noncodingCosineDistsAll[i]:
                    cosineData.append([percentage, dist, "Non-coding"])
                for dist in codingEuclideanDistsAll[i]:
                    euclideanData.append([percentage, dist, "Coding"])
                for dist in noncodingEuclideanDistsAll[i]:
                    euclideanData.append([percentage, dist, "Non-coding"])

            cosineDf = pd.DataFrame(cosineData, columns=columns)
            euclideanDf = pd.DataFrame(euclideanData, columns=columns)
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                x="Mutation Percentage",
                y="Distance",
                hue="Sequence Type",
                data=cosineDf,
                palette="Set2",
            )

            plt.xlabel("Mutation Percentage")
            plt.ylabel("Cosine Distance")
            plt.title(
                f"Comparison of Cosine Distances for Coding and Non-coding Sequences\nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length."
            )
            plt.legend(title="Sequence Type")
            plt.tight_layout()
            plt.savefig(f"{plotPath}/CosineBoxplot.png", dpi=500, bbox_inches="tight")
            # plt.show()

            plt.figure(figsize=(12, 6))
            sns.boxplot(
                x="Mutation Percentage",
                y="Distance",
                hue="Sequence Type",
                data=euclideanDf,
                palette="Set2",
            )

            plt.xlabel("Mutation Percentage")
            plt.ylabel("Euclidean Distance")
            plt.title(
                f"Comparison of Euclidean Distances for Coding and Non-coding Sequences\nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length."
            )
            plt.legend(title="Sequence Type")
            plt.tight_layout()
            plt.savefig(
                f"{plotPath}/EuclideanBoxplot.png", dpi=500, bbox_inches="tight"
            )
            # plt.show()

            """
            #For the simple chart:
            euclideanFig, euclideanAxs = plt.subplots()
            euclideanAxs.set_xlabel("Mutation Percentage")
            euclideanAxs.set_ylabel("Averaged Euclidean Distance")
            euclideanAxs.plot(mutations, codingEuclideanDistsAll, color="green")
            euclideanAxs.plot(mutations, noncodingEuclideanDistsAll, color="orange")
            euclideanAxs.legend(["Coding Sequences", "Non-coding Sequences"])

            cosineFig, cosineAxs = plt.subplots()
            cosineAxs.set_xlabel("Mutation Percentage")
            cosineAxs.set_ylabel("Averaged Cosine Distance")
            cosineAxs.plot(mutations, codingCosineDistsAll, color="green")
            cosineAxs.plot(mutations, noncodingCosineDistsAll, color="orange")
            cosineAxs.legend(["Coding Sequences", "Non-coding Sequences"])

            plt.xticks(mutations)

            euclideanFig.tight_layout(rect=[0, 0, 1, 0.93])
            euclideanFig.suptitle(
                f"Variation in averaged Euclidean Distances for the two sequence types\nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
                fontsize=10,
                y=0.98,
            )

            cosineFig.tight_layout(rect=[0, 0, 1, 0.93])
            cosineFig.suptitle(
                f"Variation in averaged Cosine Distances for the two sequence types\nModel: {model[1].split('/')[-1]}, Input sequence size: {fraction}% of maximum input length.",
                fontsize=10,
                y=0.98,
            )

            # plt.show()
            euclideanFig.savefig(
                f"{plotPath}/Euclidean.png", dpi=500, bbox_inches="tight"
            )
            cosineFig.savefig(f"{plotPath}/Cosine.png", dpi=500, bbox_inches="tight")
            """


if __name__ == "__main__":
    # seqChoice = int(input("Enter the type of sequence: Coding - 0, Non-coding  -1"))
    # seqType = types[seqChoice]
    # plotChoice = input("Enter the type of plot to be made: mutation-wise, fraction-wise, model-wise, overall")
    # plotter(types[0], "fraction-wise")
    versusPlotter()

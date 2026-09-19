import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    AffinityPropagation,
)
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
from biotite.sequence import NucleotideSequence

"""
models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
]

models = [
    "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
    "LongSafari/hyenadna-small-32k-seqlen-hf",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
]
"""
models = [
    "zhihan1996/DNABERT-S",
    "zhihan1996/DNABERT-2-117M",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "songlab/gpn-brassicales",
    "PoetschLab/GROVER",
]

genes = [
    "YAL001C",
    "YAL007C",
    "YAL018C",
    "YAL022C",
    "YAL026C",
    "YJR136C",
    "YPR192W",
]


def T_SNE(embeddings):
    # Adapted from https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/clustering_with_embeddings.ipynb#scrollTo=t-1QKCK8DHsI
    params = {"random_state": 0, "max_iter": 1000}
    tsne = TSNE(random_state=params["random_state"], max_iter=params["max_iter"])
    tsneResults = tsne.fit_transform(embeddings)
    return tsneResults, params


def U_MAP(embeddings):
    params = {
        "random_state": 42,
        "n_neighbors": 17,
        "n_epochs": None,
        # "learning_rate": 0.01,
        "min_dist": 5,
        "spread": 10,
        # "metric": "manhattan",
    }
    # Adapted from https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    reducer = umap.UMAP(**params)
    umapResults = reducer.fit_transform(embeddings)

    return umapResults, params


def plotter(model, df, method, cluster, algo=None):
    if method == "UMAP":
        results, params = U_MAP([[0]])
        columns = ["UMAP1", "UMAP2"]
    elif method == "TSNE":
        results, params = T_SNE([[0]])
        columns = ["TSNE1", "TSNE2"]

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.scatterplot(data=df, x=columns[0], y=columns[1], hue=cluster, palette="hls")
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.legend(title="Clade", fontsize=7, title_fontsize=9, bbox_to_anchor=(1, 1))
    plt.title(
        f"Scatter plot of {model}'s embeddings using {f'{algo} clustering' if cluster == 'Cluster' else method}"
    )
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    # plt.axis("equal")

    plt.show()
    save = "y"  # input(f"Save {model}'s plot? y/n")
    if save == "y":
        fig.savefig(
            f"Results/{method}/{'Clustering' if cluster == 'Cluster' else 'param testing'}/{algo if cluster == 'Cluster' else method} {' '.join([f'{key}-{value}' for key, value in params.items()])}.png"
        )


def cluster(model, embeddings, df, method, algo, param):
    # read: https://scikit-learn.org/stable/modules/clustering.html
    # sklearn clustering - note that we're clustering based on the original embeddings, not the ones whose dimesnions have been reduced:
    if algo == "KMeans":
        clusteringModel = KMeans(n_clusters=param, random_state=1).fit(embeddings)
    elif algo == "Agglomerative":
        clusteringModel = AgglomerativeClustering(
            n_clusters=param,
            # distance_threshold=param,
            linkage="average",  # with single, as expected max threshold is also around 0.065
        ).fit(embeddings)
    elif algo == "Birch":
        clusteringModel = Birch(n_clusters=30).fit(embeddings)
    elif algo == "DBSCAN":
        clusteringModel = DBSCAN(
            eps=param,
            # min_samples=30
        ).fit(
            embeddings
        )  # max eps ~= 0.065; beyond this no clusters are distinguishable. Aligns with info from single linkage hierarchical clustering!
    elif algo == "OPTICS":
        clusteringModel = OPTICS().fit(embeddings)
    elif algo == "AffinityPropagation":
        clusteringModel = AffinityPropagation().fit(embeddings)
    labels = clusteringModel.fit_predict(embeddings)
    clades = df["Clades"]
    # silScore = silhouette_score(embeddings, labels)
    # randScore = adjusted_rand_score(clades, labels)
    # replace with adjusted rand index now; have to pass the clade info too.
    df["Cluster"] = labels
    if method:
        plotter(model, df, method, "Cluster", algo)
    return randScore


def main(model, method="UMAP"):

    embeddingsSum, embeddingsAvg = [], []
    embeddings = []

    clades = []
    missing = []
    with open("../Data Labelling/labels.txt", "r+") as f:
        seqDict = eval(f.read())
        for element in seqDict:
            clades.append(
                element["clade"]
            )  # labels.txt contains 1011 strains' clade labels in sequential order, which is the same order as the embeddings files. So we can just read them in order and append to clades list.

        """
        #with the npz files:
        for seq in seqDict:
            if seq["name"] not in ids:
                missing.append(seq["name"])

        for i in ids:
            if "SACE_" + i in missing:
                ids[ids.index(i)] = "SACE_" + i

        for seq in ids:
            for element in seqDict:
                if seq == element["name"]:
                    clades.append(element["clade"])
        """

    """Full - genome embeddings:

    # This is for compressed npz files.

    model = model.split("/")[1]
    npzfile = np.load(f"../../Model Inference/Full/Hyena/{model}.npz")
    ids = sorted(npzfile.files)

    for seq in ids:
        embedding = npzfile[seq]
        embeddingsSum.append(np.sum(embedding, axis=0))
        embeddingsAvg.append(np.sum(embedding, axis=0) / len(embedding))
        embeddings.append(embedding[0])

    # this is for traditional txt/json files - single gene embeddings

    with open(
        f"../../Model Inference/Embeddings/Full/Hyena/{model}.json",  # remove hyena
        "r+",  # remember to turn back to .txt file for non-full genome embeddings
    ) as f:
        embeddingsDict = json.load(f)  # eval(f.read())
        for i in range(len(embeddingsDict["embeddingsAvg"])):
            embeddingsAvg.append(
                np.array(embeddingsDict["embeddingsAvg"][i]["embedding"])
            )
            embeddingsSum.append(
                np.array(embeddingsDict["embeddingsSum"][i]["embedding"])
            )
    """

    gene = genes[0]
    model = models[0]

    with open(
        f"../../Model Inference/Embeddings/{gene}/{(model).split('/')[1]}_{gene}.txt",  # remove Hyena
        "r+",
    ) as f:  # add hyena in a separate run.
        modelEmbeddings = eval(f.read())

    for element in modelEmbeddings:
        embeddings.append(element["embedding"])

    embeddingsSum, embeddingsAvg = np.array(embeddingsSum), np.array(embeddingsAvg)

    task = "vizualise"  # input('Enter task to be done: UMAP/TSNE Vizualisation = "vizualise", clustering vizualization = "cluster", cluster scoring = "score"')

    # method = input("Enter dimension reduction method:").upper()
    if method == "UMAP":
        results, params = U_MAP(embeddings)
        # resultsSum, paramsSum = U_MAP(embeddingsSum)
        # resultsAvg, paramsAvg = U_MAP(embeddingsAvg)
        columns = ["UMAP1", "UMAP2"]
    elif method == "TSNE":
        results, params = T_SNE(embeddings)
        # resultsSum, paramsSum = T_SNE(embeddingsSum)
        # resultsAvg, paramsAvg = T_SNE(embeddingsAvg)
        columns = ["TSNE1", "TSNE2"]

    singleGeneDF = pd.DataFrame(results, columns=columns)
    singleGeneDF["Clades"] = clades
    # sumDF = pd.DataFrame(resultsSum, columns=columns)
    # avgDF = pd.DataFrame(resultsAvg, columns=columns)
    # avgDF["Clades"] = clades
    # sumDF["Clades"] = clades

    if task == "vizualise":
        # This is for plotting the UMAP/TSNE results:
        plotter(model, singleGeneDF, method, "Clades")
        """
        print("With embeddings summed:")
        plotter(model, sumDF, method, "Clades")
        print("With embeddings averaged:")
        plotter(model, avgDF, method, "Clades")
        """

    elif task == "cluster":
        algos = [
            "KMeans",
            "Agglomerative",
            "OPTICS",
            "Birch",  # useless
            "DBSCAN",  # useless
            "AffinityPropagation",
        ]
        for algo in algos:
            cluster(model, embeddings, df, method, algo, None)
        # refactor this to handle embeddings' sum and avg

    elif task == "score":
        # plot eps vs silscore
        clusterMethod = "DBSCAN"

        fig, ax = plt.subplots()
        x, y = [], []
        n = 0.001
        while n <= 0.065:  # beyond this, it creates just a single cluster
            x.append(n)
            score = cluster(model, embeddings, df, False, clusterMethod, n)
            y.append(score)
            print(score)
            n += 0.001

        ax.plot(x, y)
        ax.set(ylim=(0, 1), xlim=(0, n))
        plt.xlabel("eps")
        plt.ylabel("Adjusted Rand Score")
        plt.title(f"{clusterMethod}")
        plt.show()

        fig.savefig(
            f"Results/Cluster Score Variation/{clusterMethod}/Adjusted Rand Score/default.png"
        )

    return f"Done with {model}"


if __name__ == "__main__":
    print(main(models[0]))

    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(main, models)
        for result in results:
            print(result)
    """

# TODO: make a summary of all the new stuff you've learnt - dimensionality reduction, different libraries anol.

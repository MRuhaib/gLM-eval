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

models = [
    "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "LongSafari/hyenadna-large-1m-seqlen-hf",
    "AIRI-Institute/gena-lm-bigbird-base-t2t",
    "songlab/gpn-brassicales",
    "PoetschLab/GROVER",
    "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    # "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
    "zhihan1996/DNABERT-S",
    "zhihan1996/DNABERT-2-117M",
]

genes = [
    ["YAL001C", 3573],
    ["YAL007C", 648],
    ["YAL018C", 978],
    ["YAL022C", 1554],
    ["YAL026C", 4068],
    ["YJR136C", 1266],
    ["YPR192W", 918],
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


if __name__ == "__main__":

    method = "UMAP"
    clades = []
    missing = []
    with open("../Data Labelling/labels.txt", "r+") as f:
        seqDict = eval(f.read())
        for element in seqDict:
            clades.append(element["clade"])

    gene = genes[0][0]

    method = "UMAP"

    fig, axs = plt.subplots(3, 3, figsize=(10, 8))
    plt.rcParams["font.size"] = 6
    plt.rcParams["axes.labelsize"] = 6
    plt.rcParams["xtick.labelsize"] = plt.rcParams["ytick.labelsize"] = 6
    count = 0

    for model in models:
        embeddings = []

        with open(
            f"../../Model Inference/Embeddings/{gene}/{(model).split('/')[1]}_{gene}.txt",  # remove Hyena
            "r+",
        ) as f:  # add hyena in a separate run.
            modelEmbeddings = eval(f.read())

        for element in modelEmbeddings:
            embeddings.append(element["embedding"])

        if method == "UMAP":
            results, params = U_MAP(embeddings)
            columns = ["UMAP1", "UMAP2"]
        elif method == "TSNE":
            results, params = T_SNE(embeddings)
            columns = ["TSNE1", "TSNE2"]

        df = pd.DataFrame(results, columns=columns)
        df["Clades"] = clades

        x_ind = count // 3
        y_ind = count % 3

        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        sns.scatterplot(
            data=df,
            x=columns[0],
            y=columns[1],
            hue="Clades",
            s=8,
            palette="Set1",
            ax=axs[x_ind][y_ind],
        )
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        # plt.legend(title="Clade", fontsize=7, title_fontsize=9, bbox_to_anchor=(1, 1))
        """
        axs[x_ind][y_ind].set_title(model.split("/")[1], fontsize=9)
        axs[x_ind][y_ind].set_xlabel(columns[0], fontsize=6)
        axs[x_ind][y_ind].tick_params(axis="both", which="major", labelsize=6)
        axs[x_ind][y_ind].tick_params(axis="both", which="minor", labelsize=5)
        axs[x_ind][y_ind].set_ylabel(columns[1], fontsize=6)
        axs[x_ind][y_ind].set_xlim(-100, 100)
        axs[x_ind][y_ind].set_ylim(-100, 100)
        axs[x_ind][y_ind].get_legend().remove()
        axs[x_ind][y_ind].set_aspect("equal", adjustable="box")
        count += 1
        """
        # plt.axis("equal")
        axs[x_ind][y_ind].set_title(model.split("/")[1], fontsize=12, pad=10)
        axs[x_ind][y_ind].set_xlabel(columns[0], fontsize=10)
        axs[x_ind][y_ind].set_ylabel(columns[1], fontsize=10)
        axs[x_ind][y_ind].tick_params(axis="both", which="major", labelsize=8)
        axs[x_ind][y_ind].set_xlim(-100, 100)
        axs[x_ind][y_ind].set_ylim(-100, 100)

        axs[x_ind][y_ind].grid(True, alpha=0.3, linewidth=0.5)

        axs[x_ind][y_ind].get_legend().remove()
        axs[x_ind][y_ind].set_aspect("equal", adjustable="box")

        for spine in axs[x_ind][y_ind].spines.values():
            spine.set_linewidth(1)
            spine.set_color("black")

        count += 1

    # plt.legend(title="Clade", fontsize=7, title_fontsize=9, bbox_to_anchor=(1, 1))
    handles, labels = axs[1][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="Clade",
        fontsize=10,
        title_fontsize=12,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),  # Positioned on the right edge
        frameon=True,
        fancybox=False,
        shadow=False,
    )
    fig.tight_layout()
    # fig.subplots_adjust(right=0.85, top=0.9, wspace=0.4, hspace=-0.3)
    fig.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)

    fig.savefig("./allUmaps3x3.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.show()

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def elbow_plot(dataset, dimension, output_plot):
    X = pd.read_csv(dataset, sep=" ", header=None).to_numpy()

    K = np.arange(1, 16)
    inertia_list = []
    for k in K:
        print("****** k = ", k)
        kmeans = KMeans(n_clusters=k, random_state=0, verbose=0).fit(X)
        inertia_list.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K, inertia_list, marker="o")
    ax.set_xlabel("Number of clusters")
    ax.set_xticks(K)
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow plot")
    plt.savefig(output_plot)


if __name__ == "__main__":
    elbow_plot(sys.argv[1], sys.argv[2], sys.argv[3])

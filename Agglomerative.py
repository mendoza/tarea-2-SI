import sys
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from generic import npArrayFromCSV


def argumentExist():
    try:
        dataset = sys.argv[1]
        clusters = int(sys.argv[2])
    except IndexError:
        print(
            "Por favor proporcione ambos valores, ingresando el dataset primero luego la cantidad de clusters"
        )
        sys.exit(1)
    return dataset, clusters


def main():
    path, clusters = argumentExist()
    X = npArrayFromCSV(path)
    # Clusters es None por mientras se prueba threshold
    agglomerative = AgglomerativeClustering(
        n_clusters=None, linkage="ward", distance_threshold=0.25)
    agglomerative.fit(X)
    x = X[:, 0]
    y = X[:, 1]
    plt.scatter(x, y, c=agglomerative.labels_)
    plt.show()


if __name__ == '__main__':
    main()

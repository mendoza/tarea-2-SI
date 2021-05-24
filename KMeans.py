import sys
from sklearn.cluster import KMeans
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
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)
    x = X[:, 0]
    y = X[:, 1]
    plt.scatter(x, y, c=kmeans.labels_)
    plt.show()

if __name__ == '__main__':
    main()

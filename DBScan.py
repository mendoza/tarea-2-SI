import sys
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from generic import npArrayFromCSV


def argumentExist():
    try:
        dataset = sys.argv[1]
        eps = float(sys.argv[2])
        min_samples = float(sys.argv[3])
    except IndexError:
        print(
            "Por favor proporcione ambos valores, ingresando el dataset primero luego eps y termina con el min_samples"
        )
        sys.exit(1)
    return dataset, eps, min_samples


path, eps, min_samples = argumentExist()
X = npArrayFromCSV(path)
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)
x = X[:, 0]
y = X[:, 1]
plt.scatter(x, y, c=dbscan.labels_)
plt.show()

import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
from generic import dataFrame


def argumentExist():
    try:
        trainingPath = sys.argv[1]
        testingPath = sys.argv[2]
        k = int(sys.argv[3])
    except IndexError:
        print(
            "Por favor proporcione ambos valores, ingresando el training primero luego el testing y al final k"
        )
        sys.exit(1)
    return trainingPath, testingPath, k


def proccessData(df):
    x = {'si': 1, 'no': 0, 'A': 1, 'B': 2, 'C': 3,
         '30-80': 1, '80-120': 2, '120+': 3, 'real': 1, 'ficticia': 0, 'lineal': 1, 'mosaico': 2, 'circulo': 3, 'contemporaneo': 1, 'futuro': 2, 'pasado': 3, 'simple': 1, 'compleja': 0, 'horror': 1, 'accion': 2, 'comedia': 3, 'drama': 4}

    for col in df.columns:
        df[col] = df[col].map(x)
    return df


def datasetToStandard(df):
    clase = df.columns[-1]
    y = df[clase]
    x = df.iloc[:, :-1]
    return x, y


def main():
    trainingPath, testingPath, k = argumentExist()
    training = proccessData(dataFrame(trainingPath))
    testing = proccessData(dataFrame(testingPath))
    x, y = datasetToStandard(training)
    testingX, testingY = datasetToStandard(testing)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x.values, y.values)
    # print(neigh.predict([testingX.values[1]]))
    # print(testingY.values[1])


if __name__ == '__main__':
    main()

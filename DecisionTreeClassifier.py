import sys
from numpy.core.numeric import indices
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from generic import dataFrame
import pandas as pd


def argumentExist():
    try:
        trainingPath = sys.argv[1]
        testingPath = sys.argv[2]
        criterio = sys.argv[3]
        max_depth = int(sys.argv[4])
    except IndexError:
        print(
            "Por favor proporcione ambos valores, ingresando el training primero luego el testing y al final k"
        )
        sys.exit(1)
    return trainingPath, testingPath, criterio, max_depth


def proccessData(df):
    x = {'si': 1, 'no': 0, 'A': 1, 'B': 2, 'C': 3,
         '30-80': 1, '80-120': 2, '120+': 3, 'real': 1, 'ficticia': 0, 'lineal': 1, 'mosaico': 2, 'circulo': 3, 'contemporaneo': 1, 'futuro': 2, 'pasado': 3, 'simple': 1, 'compleja': 0, 'horror': 0, 'accion': 1, 'comedia': 2, 'drama': 3}

    for col in df.columns:
        df[col] = df[col].map(x)
    return df


def datasetToXY(df):
    clase = df.columns[-1]
    y = df[clase]
    x = df.iloc[:, :-1]
    return x, y


def main():
    trainingPath, testingPath, criterio, max_depth = argumentExist()

    # Dataframes from csv
    training = proccessData(dataFrame(trainingPath))
    testing = proccessData(dataFrame(testingPath))

    x, y = datasetToXY(training)

    testingX, testingY = datasetToXY(testing)

    classifier = DecisionTreeClassifier(
        criterion=criterio, max_depth=max_depth)

    classifier.fit(x, y)

    y_pred = classifier.predict(testingX)

    # print(y_pred)

    print(confusion_matrix(testingY, y_pred))
    print(classification_report(testingY, y_pred))

    # classes = ['horror', 'accion', 'comedia', 'drama']
    # for j in range(len(classes)):
    #     indixes = []
    #     for i in range(len(testingX)):
    #         if testingY.values[i] == j:
    #             indixes.append(i)
    #     X = []
    #     Y = []
    #     for i in indixes:
    #         X.append(testingX.values[i])
    #         Y.append(testingY.values[i])

    # pred = neigh.predict(X)
    # print(f"Haciendo la evaluacion de la clase {classes[j]}")
    # print("Recall:", recall_score(Y, pred, average="weighted",zero_division=0))
    # print("F1 Score:", f1_score(Y, pred, average="weighted"))
    # print("Accuracy:", accuracy_score(Y, pred))


if __name__ == '__main__':
    main()

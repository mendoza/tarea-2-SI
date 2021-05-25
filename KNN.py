import sys
from numpy.core.numeric import indices
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
from generic import dataFrame
import pandas as pd


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
    df2 = pd.DataFrame({
        'animada': [],
        'basada_libro': [],
        'A': [],
        'B': [],
        'C': [],
        'desenlace_feliz': [],
        '30-80':  [],
        '80-120': [],
        '120+': [],
        'lineal': [],
        'mosaico': [],
        'circulo': [],
        'real': [],
        'ficticia': [],
        'saga': [],
        'contemporaneo': [],
        'futuro': [],
        'pasado': [],
        'simple': [],
        'compleja': [],
        'class': []
    })

    classes = ['horror', 'accion', 'comedia', 'drama']
    for row in range(len(df.values)):
        new_row = {
            'animada': 0,
            'basada_libro': 0,
            'A': 0,
            'B': 0,
            'C': 0,
            'desenlace_feliz': 0,
            '30-80':  0,
            '80-120': 0,
            '120+': 0,
            'lineal': 0,
            'mosaico': 0,
            'circulo': 0,
            'real': 0,
            'ficticia': 0,
            'saga': 0,
            'contemporaneo': 0,
            'futuro': 0,
            'pasado': 0,
            'simple': 0,
            'compleja': 0,
            'class': 0
        }
        for header in df.columns:
            if header in new_row.keys():
                if header == 'class':
                    new_row[header] = classes.index(df[header].iloc[row])
                    continue
                new_row[header] = 1 if df[header].iloc[row] == 'si' else 0
            else:
                name = df[header].iloc[row]
                new_row[name] = 1
        df2 = df2.append(new_row, ignore_index=True)
    return df2


def datasetToXY(df):
    clase = df.columns[-1]
    y = df[clase]
    x = df.iloc[:, :-1]
    return x, y


def main():
    trainingPath, testingPath, k = argumentExist()
    training = proccessData(dataFrame(trainingPath))
    testing = proccessData(dataFrame(testingPath))

    x, y = datasetToXY(training)

    testingX, testingY = datasetToXY(testing)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x.values, y.values)

    classes = ['horror', 'accion', 'comedia', 'drama']
    for j in range(len(classes)):
        indixes = []
        for i in range(len(testingX)):
            if testingY.values[i] == j:
                indixes.append(i)
        X = []
        Y = []
        for i in indixes:
            X.append(testingX.values[i])
            Y.append(testingY.values[i])

        pred = neigh.predict(X)
        print(f"Haciendo la evaluacion de la clase {classes[j]}")
        print("Recall:", recall_score(Y, pred, average="weighted",zero_division=0))
        print("F1 Score:", f1_score(Y, pred, average="weighted"))
        print("Accuracy:", accuracy_score(Y, pred))


if __name__ == '__main__':
    main()

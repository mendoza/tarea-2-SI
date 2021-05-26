import sys
import time
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
from generic import dataFrame
import pandas as pd


def argumentExist():
    try:
        trainingPath = sys.argv[1]
        testingPath = sys.argv[2]
        criterio = sys.argv[3]
        max_depth = int(sys.argv[4]) if sys.argv[4] != 'None' else None
    except IndexError:
        print(
            "Por favor proporcione todos los valores, ingresando el training primero, luego el testing, luego el criterio y para terminar el depth maximo"
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
    start = time.time()
    y_pred = classifier.predict(testingX)
    tiempo = f'Tiempo: {time.time() - start}'

    # print(y_pred)
    tree_rules = export_text(classifier, feature_names=list(x.columns))
    print(tree_rules)
    print(tiempo)
    # print(confusion_matrix(testingY, y_pred))
    print(classification_report(testingY, y_pred, zero_division=0, digits=4))


if __name__ == '__main__':
    main()

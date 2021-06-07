import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error
from generic import dataFrame, datasetToXY


def proccessData(df):
    mapa = {'si': 1, 'no': 0, 'A': 1, 'B': 2, 'C': 3,
            '30-80': 1, '80-120': 2, '120+': 3, 'real': 1, 'ficticia': 0, 'lineal': 1, 'mosaico': 2, 'circulo': 3, 'contemporaneo': 1, 'futuro': 2, 'pasado': 3, 'simple': 1, 'compleja': 0, 'horror': 0, 'accion': 1, 'comedia': 2, 'drama': 3}

    for col in df.columns:
        df[col] = df[col].map(mapa)
    return df


def main():
    trainingPath = 'genero_peliculas_training.csv'
    training = proccessData(dataFrame(trainingPath))
    x_train, y_train = datasetToXY(training)

    testPath = 'genero_peliculas_testing.csv'
    testing = proccessData(dataFrame(testPath))
    x_test, y_test = datasetToXY(testing)

    tree = RandomForestClassifier(n_estimators=100,
                                  criterion="gini", max_depth=5, max_features="auto")
    tree.fit(x_train, y_train)

    print(classification_report(y_test, tree.predict(x_test)))
    print("MSE test:", mean_squared_error(y_test, tree.predict(x_test)))
    print("MSE train:", mean_squared_error(y_train, tree.predict(x_train)))


if __name__ == '__main__':
    main()

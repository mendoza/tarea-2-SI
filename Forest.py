import json
import pandas as pd
from scipy.sparse import data
from scipy.sparse.construct import rand
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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

    testPath = 'genero_peliculas_testing.csv'
    testing = proccessData(dataFrame(testPath))

    params = {
        'criterion': {
            'default': 'gini',
            'values': ['gini', 'entropy']
        },
        'n_estimators': {
            'default': 100,
            'values': [10, 20, 30, 40, 50]
        },
        'max_features': {
            'default': "auto",
            'values': ["auto", "sqrt", "log2"]
        },
        'max_depth': {
            'default': None,
            'values': [5, 10, 15, 20, 25]
        }
    }
    x_train, y_train = datasetToXY(training)

    x_test, y_test = datasetToXY(testing)
    dicts = []
    combinations = []
    for key in params.keys():
        for inner in params[key]['values']:
            n_estimators = params['n_estimators']['default'] if key != 'n_estimators' else inner
            criterion = params['criterion']['default'] if key != 'criterion' else inner
            max_depth = params['max_depth']['default'] if key != 'max_depth' else inner
            max_features = params['max_features']['default'] if key != 'max_features' else inner

            forest = RandomForestClassifier(
                n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_features=max_features)
            forest.fit(x_train, y_train)

            report = classification_report(
                y_test, forest.predict(x_test), output_dict=True)

            report['combination'] = {
                "arboles": str(n_estimators),
                "criterio": criterion,
                "profundidad": str(max_depth),
                "caracteristicas": max_features}

            dicts.append(report)

    df = pd.DataFrame(
        {}, columns=['conf_id', 'f1-0', 'f1-1', 'f1-2', 'f1-3', 'arboles', 'criterio', 'profundidad', 'caracteristicas'])
    for dicty in dicts:
        df = df.append({'conf_id': dicts.index(
            dicty), 'f1-0': dicty["0"]['f1-score'], 'f1-1': dicty["1"]['f1-score'], 'f1-2': dicty["2"]['f1-score'], 'f1-3': dicty["3"]['f1-score'],
            "arboles": dicty['combination']["arboles"],
            "criterio": dicty['combination']["criterio"],
            "profundidad": dicty['combination']["profundidad"],
            "caracteristicas": dicty['combination']["caracteristicas"]}, ignore_index=True)
    df.to_csv('metrics.csv', index=False)


if __name__ == '__main__':
    main()

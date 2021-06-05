import sys
import time
from scipy.sparse.construct import rand
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from generic import dataFrame


def mapear(mapa, df):
    for col in df.columns:
        df[col] = df[col].map(mapa)
    return df


def main():
    pathTraining = 'seguros_training_data.csv'
    X = dataFrame(pathTraining)
    Y = X[X.columns[-1]].map({'plan_C': 1, 'plan_B': 0})
    X = mapear({'Si': 1, 'No': 0}, X.iloc[:, :-1])

    pathTesting = 'seguros_testing_data.csv'
    x = dataFrame(pathTesting)
    y = x[x.columns[-1]].map({'plan_C': 1, 'plan_B': 0})
    x = mapear({'Si': 1, 'No': 0}, x.iloc[:, :-1])

    reg = LogisticRegression(random_state=0)
    reg.fit(X, Y)

    start = time.time()
    pred_y = reg.predict(x)
    print(f'Time: {time.time() - start}')
    print(classification_report(y, pred_y))


if __name__ == '__main__':
    main()

import time
from scipy.sparse.construct import rand
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error
from generic import dataFrame, datasetToXY


def mapear(mapa, df):
    for col in df.columns:
        df[col] = df[col].map(mapa)
    return df


def main():
    trainingPath = 'seguros_training_data.csv'
    x_train, y_train = datasetToXY(dataFrame(trainingPath))
    y_train = y_train.map({'plan_C': 1, 'plan_B': 0})
    x_train = mapear({'Si': 1, 'No': 0}, x_train)

    testPath = 'seguros_testing_data.csv'
    x_test, y_test = datasetToXY(dataFrame(testPath))
    y_test = y_test.map({'plan_C': 1, 'plan_B': 0})
    x_test = mapear({'Si': 1, 'No': 0}, x_test)

    log = LogisticRegression()
    log.fit(x_train, y_train)

    start = time.time()
    y_pred = log.predict(x_test)
    print(f'Time: {time.time() - start}')
    print(log.coef_)
    print(abs(log.coef_))
    print(max(log.coef_[0]))
    print(min(log.coef_[0]))
    print(mean_squared_error(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()

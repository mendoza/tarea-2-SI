from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from generic import dataFrame, datasetToXY


def main():
    trainingPath = 'regression_train.csv'
    x_train, y_train = datasetToXY(dataFrame(trainingPath))

    testPath = 'regression_test.csv'
    x_test, y_test = datasetToXY(dataFrame(testPath))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    lin = LinearRegression()
    lin.fit(x_train, y_train)

    print(mean_squared_error(y_test, lin.predict(x_test)))
    print(mean_squared_error(y_train, lin.predict(x_train)))
    print(lin.coef_)
    print(lin.intercept_)


if __name__ == '__main__':
    main()

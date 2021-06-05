from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from generic import dataFrame, datasetToXY


def main():
    trainingPath = 'regression_train.csv'
    X, Y = datasetToXY(dataFrame(trainingPath))

    testPath = 'regression_test.csv'
    x, y = datasetToXY(dataFrame(testPath))

    lin = Lasso()
    lin.fit(X, Y)

    print(mean_squared_error(y, lin.predict(x)))
    print(lin.coef_)
    print(lin.intercept_)


if __name__ == '__main__':
    main()

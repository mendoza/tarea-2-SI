import sys
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from generic import dataFrame


def main():
    pathTraining = 'regression_train.csv'
    X = dataFrame(pathTraining)
    Y = X[X.columns[-1]]
    X = X.iloc[:, :-1]

    pathTesting = 'regression_test.csv'
    x = dataFrame(pathTesting)
    y = x[x.columns[-1]]
    x = x.iloc[:, :-1]

    reg = LinearRegression(normalize=True)
    reg.fit(X, Y)

    print(mean_squared_error(reg.predict(x), y))
    print(reg.coef_)
    print(reg.intercept_)


if __name__ == '__main__':
    main()

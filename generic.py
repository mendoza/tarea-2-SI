import numpy as np
import pandas as pd


def npArrayFromCSV(path):
    csvfile = pd.read_csv(path, encoding='utf-8')
    return np.array(csvfile.values)


def dataFrame(path):
    csvfile = pd.read_csv(path, encoding='utf-8')
    return csvfile

def datasetToXY(df):
    clase = df.columns[-1]
    y = df[clase]
    x = df.iloc[:, :-1]
    return x, y

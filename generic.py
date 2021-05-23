import numpy as np
import pandas as pd

def npArrayFromCSV(path):
    csvfile = pd.read_csv(path, encoding='utf-8')
    return np.array(csvfile.values)
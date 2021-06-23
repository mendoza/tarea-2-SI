import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('datos.csv')
average = dataframe['avg'].to_list()
maximo = dataframe['max'].to_list()
minimo = dataframe['min'].to_list()
plt.title('MSE por epoca')
plt.plot(average, label="Promedio por epoca")
plt.plot(maximo, label="Maximo por epoca")
plt.plot(minimo, label="Minimo por epoca")
plt.legend(loc="upper left")
plt.show()

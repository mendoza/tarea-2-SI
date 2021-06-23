import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm


def argumentExist():
    try:
        csv_file = sys.argv[1]
        epochs = int(sys.argv[2])
        return csv_file, epochs
    except IndexError:
        print(
            "Por favor proporcione los parametros, ingresando el archivo de caracteristicas primero luego la entrada del etiquetas y como ultimo la salida del clasificador"
        )
        sys.exit(1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layers
        self.l1 = nn.Linear(2, 2)
        nn.init.uniform_(self.l1.weight, -1/sqrt(5), 1/sqrt(5))
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(2, 2)
        nn.init.uniform_(self.l2.weight, -1/sqrt(5), 1/sqrt(5))

    def forward(self, x):
        # linear layers
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)
        # out = self.sigmoid(out)
        return out


class DatasetBomnito(Dataset):
    def __init__(self, csv_file, root_dir):
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.classes = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = list(self.dataset.iloc[idx])
        entrada = row[:2]
        salida = self.classes.index(row[2:])
        return torch.tensor(entrada), torch.tensor(salida)


def trainEpoch(dataloader):
    models = [Net() for _ in range(20)]
    losses = [nn.MSELoss() for _ in range(20)]
    optimizeres = [optim.SGD(model.parameters(), lr=0.2) for model in models]
    mses = []
    for i, model in enumerate(models):
        train_loss = 0
        runnning_mse = 0
        # training steps
        for batch_index, (data, target) in enumerate(dataloader):
            optimizeres[i].zero_grad()
            output = model(data.float())
            loss = losses[i](output, target.float())
            loss.backward()
            optimizeres[i].step()
            train_loss += loss.item()*data.size(0)
            squared_error = ((output - target) *
                             (output - target)).sum().data
            runnning_mse += squared_error

        mse = sqrt(runnning_mse/len(dataloader))
        mses.append(mse)
    return mses


def main():
    csv_file, epochs = argumentExist()
    data = DatasetBomnito(csv_file, root_dir='')
    dataloader = DataLoader(data, batch_size=2, shuffle=False)
    torch.no_grad()
    dataframe = pd.DataFrame(
        {"epoch": [], "max": [], "min": [], "avg": []}, columns=["epoch", "max", "min", "avg"])
    for i in tqdm(range(epochs)):
        mses = trainEpoch(dataloader)
        epoch = i
        maximo = max(mses)
        minimo = min(mses)
        average = sum(mses)/len(mses)

        dataframe = dataframe.append(
            {"epoch": epoch, "max": maximo, "min": minimo, "avg": average}, ignore_index=True)
    dataframe.to_csv('datos1.csv', index=False)


if __name__ == '__main__':
    main()

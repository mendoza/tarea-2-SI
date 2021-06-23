import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report


def argumentExist():
    try:
        csv_training = sys.argv[1]
        epochs = int(sys.argv[2])
        csv_valid = sys.argv[3]
        epsilon = float(sys.argv[4])
        max_epochs = int(sys.argv[5])
        return csv_training, epochs, csv_valid, epsilon, max_epochs
    except IndexError:
        print(
            "Por favor proporcione los parametros, ingresando el archivo de caracteristicas primero luego la entrada del etiquetas y como ultimo la salida del clasificador"
        )
        sys.exit(1)


class Net(nn.Module):
    def __init__(self, entrada, mid, salida):
        super(Net, self).__init__()
        # linear layers
        self.l1 = nn.Linear(entrada, mid)
        nn.init.uniform_(self.l1.weight, -1/sqrt(5), 1/sqrt(5))
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(mid, salida)
        nn.init.uniform_(self.l2.weight, -1/sqrt(5), 1/sqrt(5))
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # linear layers
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)
        out = self.lsm(out)
        return out


class DatasetBomnito(Dataset):
    def __init__(self, csv_file, root_dir):
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = list(self.dataset.iloc[idx])
        row = [1.0 if item == 'Si' or item ==
               'plan_C' else 0.0 for item in row]
        entrada = row[:-1]
        salida = int(row[-1])
        return torch.tensor(entrada), torch.tensor(salida, dtype=torch.long)


def main():
    csv_training, epochs, csv_valid, epsilon, max_epochs = argumentExist()
    training_data = DatasetBomnito(csv_training, root_dir='')
    valid_data = DatasetBomnito(csv_valid, root_dir='')
    training_loader = DataLoader(training_data, batch_size=2, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=2, shuffle=False)
    torch.no_grad()
    model = Net(10, 4, 2)
    fails = 0
    minimum_validation_loss = np.inf
    for epoch in range(epochs):
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.2)
        train_loss = 0
        valid_loss = 0
        runnning_mse = 0
        model.train()
        for batch_index, (data, target) in enumerate(training_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            squared_error = ((output - target) *
                             (output - target)).sum().data
            runnning_mse += squared_error

        model.eval()
        for batch_index, (data, target) in enumerate(valid_loader):
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
        mse = sqrt(runnning_mse/len(training_loader))
        print(
            f'Epoch {epoch}\t Training Loss: {train_loss/len(training_loader)}\t Validation Loss:{valid_loss/len(valid_loader)}\t MSE: {mse}')
        print(fails)
        if valid_loss <= minimum_validation_loss:
            print(
                f'Validation loss decreased from {round(minimum_validation_loss, 6)} to {round(valid_loss, 6)}')
            minimum_validation_loss = valid_loss
            fails = 0
        else:
            fails += 1
            if fails >= max_epochs:
                break

    test_data = DatasetBomnito("seguros_training_data.csv", root_dir='')
    entradas = []
    salidas = []
    for (entrada, salida) in test_data:
        entradas.append(entrada)
        salidas.append(salida)
    tensor_in = torch.stack(entrada)
    tensor_out = torch.stack(salidas)
    print(model(tensor_in))


if __name__ == '__main__':
    main()

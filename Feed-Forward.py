import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        nn.init.uniform_(self.l1.weight, -1/sqrt(5), 1/sqrt(5))
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes, dtype=torch.float64)
        nn.init.uniform_(self.l2.weight, -1/sqrt(5), 1/sqrt(5))

    def forward(self, x):
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)

        # no activation and no softmax at the end
        return out


def to_tensor(arr):
    return torch.tensor(arr)


def main():
    model = NeuralNet(2, 2, 2)
    model.eval()
    n1 = np.asarray([0, 0], dtype=np.float64)
    n2 = np.asarray([0, 1], dtype=np.float64)
    n3 = np.asarray([1, 0], dtype=np.float64)
    n4 = np.asarray([1, 1], dtype=np.float64)

    print(model(to_tensor(n1)))
    print(model(to_tensor(n2)))
    print(model(to_tensor(n3)))
    print(model(to_tensor(n4)))
    torch.save(model.state_dict(), "pesos_parte_1.pkl")


if __name__ == '__main__':
    main()

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, rotation_representation_dim, width, height):
        super(CNN, self).__init__()
        Z_DIM = rotation_representation_dim
        IMAGE_CHANNEL = 3
        Z_DIM = 10
        G_HIDDEN = 64
        X_DIM = 64
        D_HIDDEN = 64

        self.INP_SIZE = 5
        self.rotation_representation_dim = rotation_representation_dim
        self.inp = nn.Linear(self.rotation_representation_dim, self.INP_SIZE * self.INP_SIZE * 10)
        self.seq = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 2, IMAGE_CHANNEL, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.inp(x)
        x = self.seq(x.reshape(-1, 10, self.INP_SIZE, self.INP_SIZE))
        return x.permute(0, 2, 3, 1)

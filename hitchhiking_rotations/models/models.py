import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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

    def forward(self, x):
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


class MLPNetPCD(nn.Module):

    def __init__(self, in_size, out_size):
        super(MLPNetPCD, self).__init__()

        self.LR = nn.LeakyReLU(0.3)
        self.net = nn.Sequential(
            nn.Conv1d(in_size[0], 64, kernel_size=1),
            self.LR,
            nn.Conv1d(64, 128, kernel_size=1),
            self.LR,
            nn.Conv1d(128, 256, kernel_size=1),
            self.LR,
            nn.Conv1d(256, 1024, kernel_size=1),
            self.LR,
            nn.MaxPool1d(kernel_size=in_size[1]),

            nn.Flatten(),

            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            self.LR,
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            self.LR,

            nn.Linear(512, out_size),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        self.net.apply(init_weights)

    def forward(self, x):
        out = self.net(x)
        return out
# ------------------------------------------------------------------------------


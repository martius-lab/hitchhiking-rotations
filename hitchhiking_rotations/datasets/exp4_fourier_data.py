import os

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR

import matplotlib.pyplot as plt

class FourierDataset(Dataset):
    """
    Loads data from fourier dataset
    """
    def __init__(self, device, basis_num, func_num):

        path = f"{HITCHHIKING_ROOT_DIR}/assets/datasets/exp4_data"
        input_path = f'{path}/inputs_basis{basis_num}-func{func_num}.npy'
        output_path = f'{path}/outputs_basis{basis_num}-func{func_num}.npy'
        self.in_rot = torch.from_numpy(np.load(input_path).astype(np.float32)).to(self.device)
        self.out_feature = torch.from_numpy(np.load(output_path).astype(np.float32)).to(self.device)

        self.device = device
        self.N = len(self.out_feature)
        self.ixs = np.arange(self.N)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.in_rot[idx], self.out_feature[idx]

class random_fourier_function():

    def __init__(self, n_basis, A0=0., L=1.):
        self.L = L
        self.n_basis = n_basis
        self.A0 = A0
        self.A = np.random.normal(size=n_basis)
        self.B = np.random.normal(size=n_basis)
        self.matrix = np.random.normal(size=(1, 9))

    def __call__(self, x):
        fFs = self.A0 / 2
        for k in range(len(self.A)):
            fFs = (fFs + self.A[k] * np.cos((k + 1) * np.pi * np.matmul(self.matrix, x) / self.L) +
                   self.B[k] * np.sin((k + 1) * np.pi * np.matmul(self.matrix, x) / self.L))
        return fFs

def plot_fourier_data(rotations, features):
    import pandas as pd
    import seaborn as sns

    data = np.c_[rotations, features]
    df = pd.DataFrame(data)
    sns.set(style="ticks")

    g = sns.PairGrid(df, diag_sharey=True)

    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    g.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    plt.show()

def create_dataset(N_points=2000, N_basis=7, N_func=20, key=42):

    path = f"{HITCHHIKING_ROOT_DIR}/assets/datasets/exp4_data"
    print(f"Data stored in: {path}")

    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(key)
    for i in range(N_basis):
        for j in range(N_func):
            in_rot = Rotation.random(N_points)
            inputs = in_rot.as_matrix().reshape(N_points, -1)
            four_func = random_fourier_function(N_basis)
            out_features = np.apply_along_axis(four_func, 1, inputs)

            #plot_fourier_data(inputs, out_features)

            np.save(path + f'/inputs_basis{i}-func{j}.npy', in_rot)
            np.save(path + f'/outputs_basis{i}-func{j}.npy', out_features)

            data_info = {'N_points': N_points, 'N_basis': N_basis, 'N_func': N_func}
            np.save(path + f'/data_info.npy', data_info)

if __name__ == "__main__":
    create_dataset(N_points=100, N_basis=2, N_func=5, key=42)
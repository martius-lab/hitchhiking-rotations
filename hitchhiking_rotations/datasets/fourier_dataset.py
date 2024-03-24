import os
from os.path import join

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import roma
import pandas as pd
import seaborn as sns

from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle, load_pickle


class PoseToFourierDataset(Dataset):
    """
    Loads data from fourier dataset
    """

    def __init__(self, mode, dataset_size, device, nb, nf):
        path = join(
            HITCHHIKING_ROOT_DIR, "assets", "datasets", "fourier_dataset", f"fourier_dataset_{mode}_nb{nb}_nf{nf}.pkl"
        )

        if os.path.exists(path):
            dic = load_pickle(path)
            quats, features = dic["quats"], dic["features"]
            print(f"Loaded fourier_dataset_{mode}_nb{nb}_nf{nf}.pkl: {path}")
        else:
            quats, features = create_data(N_points=dataset_size, nb=nb, seed=nf)
            dic = {"quats": quats, "features": features}
            save_pickle(dic, path)
            print(f"Saved fourier_dataset_{mode}_nb{nb}_nf{nf}.pkl: {path}")

        self.features = torch.from_numpy(features).to(device)
        self.quats = torch.from_numpy(quats).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return roma.unitquat_to_rotmat(self.quats[idx]).type(torch.float32), self.features[idx]


class random_fourier_function:
    def __init__(self, n_basis, seed, L=np.pi):
        np.random.seed(seed)
        self.n_basis = n_basis
        self.L = L
        self.A = np.random.normal(size=n_basis)
        self.B = np.random.normal(size=n_basis)
        self.matrix = np.random.normal(size=(1, 9))

    def __call__(self, x):
        fFs = 0.0
        for k in range(len(self.A)):
            input = np.matmul(self.matrix, x)
            fFs += self.A[k] * np.cos((k + 1) * np.pi * input / self.L) + self.B[k] * np.sin(
                (k + 1) * np.pi * input / self.L
            )
        return fFs

    def input_to_fourier(self, x):
        return np.matmul(self.matrix, x)


def batch_normalize(arr):
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std[std == 0] = 1
    return (arr - mean) / std


def create_data(N_points, nb, seed):
    """
    Create data from fourier series.
    Args:
        N_points: Number of random rotations to generate
        nb: Number of fourier basis that form the target function
        seed: Used to randomly initialize fourier function coefficients
    Returns:
        rots: Random rotations
        features: Target function evaluated at rots
    """
    np.random.seed(seed)
    rots = Rotation.random(N_points)
    inputs = rots.as_matrix().reshape(N_points, -1)
    four_func = random_fourier_function(nb, seed)
    features = np.apply_along_axis(four_func, 1, inputs)

    features = batch_normalize(features)
    return rots.as_quat().astype(np.float32), features.astype(np.float32)


def plot_fourier_data(rotations, features):
    """Plot distribution of rotations and features."""
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


def plot_fourier_func(nb, seed):
    """Plot the target function."""
    rots = Rotation.random(400)
    inputs = rots.as_matrix().reshape(400, -1)
    four_func = random_fourier_function(nb, seed)
    four_in = np.apply_along_axis(four_func.input_to_fourier, 1, inputs)
    features = np.apply_along_axis(four_func, 1, inputs)
    features2 = batch_normalize(features)
    sorted_indices = np.argsort(four_in, axis=0)

    plt.figure()
    plt.plot(four_in[sorted_indices].flatten(), features[sorted_indices].flatten(), linestyle="-", marker=None)
    plt.plot(
        four_in[sorted_indices].flatten(), features2[sorted_indices].flatten(), linestyle="-", color="red", marker=None
    )
    plt.title(f"nb: {nb}, seed: {seed},\n matrix: {four_func.matrix}")
    plt.show()


if __name__ == "__main__":
    # Analyze created data
    for b in range(1, 6):
        for s in range(0, 3):
            # rots, features = create_data(N_points=100, nb=b, seed=s)
            # data_stats(rots, features)
            # plot_fourier_data(rots, features)
            plot_fourier_func(b, s)

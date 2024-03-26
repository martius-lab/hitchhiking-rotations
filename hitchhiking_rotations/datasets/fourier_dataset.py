import os
from os.path import join

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import roma
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_default_device", jax.devices("cpu")[0])

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


def random_fourier_function(x, nb, seed):
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    model = eqx.nn.MLP(in_size=9, out_size=1, width_size=50, depth=1, key=key1)
    A = jax.random.normal(key=key2, shape=(nb,))
    B = jax.random.normal(key=key3, shape=(nb,))

    input = model(x)
    fFs = 0.0
    for k in range(len(A)):
        fFs += A[k] * jnp.cos((k + 1) * jnp.pi * input) + B[k] * jnp.sin((k + 1) * jnp.pi * input)
    return fFs


def input_to_fourier(x, seed):
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    model = eqx.nn.MLP(in_size=9, out_size=1, width_size=50, depth=1, key=key1)
    return model(x)


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
        seed: Used to randomly initialize fourier function
    Returns:
        rots: Random rotations
        features: Target function evaluated at rots
    """
    np.random.seed(seed)
    rots = Rotation.random(N_points)
    inputs = rots.as_matrix().reshape(N_points, -1)
    features = np.array(jax.vmap(random_fourier_function, in_axes=[0, None, None])(inputs, nb, seed).reshape(-1, 1))
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
    four_in = np.array(jax.vmap(input_to_fourier, [0, None])(inputs, seed))
    features = np.array(jax.vmap(random_fourier_function, [0, None, None])(inputs, nb, seed))
    features2 = batch_normalize(features)
    sorted_indices = np.argsort(four_in, axis=0)

    plt.figure()
    plt.plot(four_in[sorted_indices].flatten(), features[sorted_indices].flatten(), linestyle="-", marker=None)
    plt.plot(
        four_in[sorted_indices].flatten(), features2[sorted_indices].flatten(), linestyle="-", color="red", marker=None
    )
    plt.title(f"nb: {nb}, seed: {seed}")
    plt.show()


if __name__ == "__main__":
    # Analyze created data
    for b in range(1, 7):
        for s in range(0, 6):
            # rots, features = create_data(N_points=100, nb=b, seed=s)
            # data_stats(rots, features)
            # plot_fourier_data(rots, features)
            print("MLP PyTree used to create Fourier function inputs:")
            model = eqx.nn.MLP(in_size=9, out_size=1, width_size=50, depth=1, key=jax.random.PRNGKey(42))
            eqx.tree_pprint(model)
            plot_fourier_func(b, s)

#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import os
import yaml
import pickle
from omegaconf import OmegaConf

__all__ = ["file_path", "load_yaml", "load_pickle", "save_pickle", "save_omega_cfg"]


def file_path(string: str) -> str:
    """Checks if string is a file path

    Args:
        string (str): Potential file path

    Raises:
        NotADirectoryError: String is not a fail path

    Returns:
        (str): Returns the file path
    """

    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_yaml(path: str) -> dict:
    """Loads yaml file

    Args:
        path (str): File path

    Returns:
        (dict): Returns content of file
    """
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    if res is None:
        res = {}
    return res


def load_pickle(path: str) -> dict:
    """Load pickle file
    Args:
        path (str): File path
    Returns:
        (dict): Returns content of file
    """

    with open(path, "rb") as file:
        res = pickle.load(file)
    return res


def save_pickle(cfg, path: str):
    """Saves to pickle file

    Args:
        cfg (dict): Configuration
        path (str): File path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(cfg, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_omega_cfg(cfg, path):
    """
    Args:
        cfg (omegaconf): Cfg file
        path (str): File path
    """
    with open(path, "rb") as file:
        OmegaConf.save(config=cfg, f=file)

#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle, load_pickle
import os
from os.path import join
from scipy.spatial.transform import Rotation
import torch
import roma
from torch.utils.data import Dataset


class CubeImageToPoseDataset(Dataset):
    def __init__(self, mode, dataset_size, device):
        rots = Rotation.random(dataset_size)
        quats = rots.as_quat()

        self.quats = torch.from_numpy(quats)
        self.imgs = []

        path = join(HITCHHIKING_ROOT_DIR, "assets", "datasets", "cube_dataset", f"cube_dataset_{mode}.pkl")

        if os.path.exists(path):
            dic = load_pickle(path)
            self.imgs, self.quats = dic["imgs"], dic["quats"]
            print(f"Cube-Dataset {mode}-file loaded: {path}")
        else:
            from .cube_data_generator import CubeDataGenerator

            dg = CubeDataGenerator(height=64, width=64)
            for i in range(dataset_size):
                # TODO normalize data
                self.imgs.append(torch.from_numpy(dg.render_img(quats[i])))
            dic = {"imgs": self.imgs, "quats": self.quats}
            save_pickle(dic, path)
            print(f"Cube-Dataset {mode}-file created and saved: {path}")

        self.imgs = [i.to(device) for i in self.imgs]
        self.quats = self.quats.to(device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx].type(torch.float32) / 255, roma.unitquat_to_rotmat(self.quats[idx]).type(torch.float32)


class PoseToCubeImageDataset(CubeImageToPoseDataset):
    def __init__(self, mode, dataset_size, device):
        super(PoseToCubeImageDataset, self).__init__(mode, dataset_size, device)

    def __getitem__(self, idx):
        return roma.unitquat_to_rotmat(self.quats[idx]).type(torch.float32), self.imgs[idx].type(torch.float32) / 255


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    dataset = CubeImageToPoseDataset("train", 2048, "cpu")
    for i in range(10):
        img, quat = dataset[i]
        img = Image.fromarray(np.uint8(img.cpu().numpy() * 255))
        img.save(join(HITCHHIKING_ROOT_DIR, "results", f"example_img_{i}.png"))

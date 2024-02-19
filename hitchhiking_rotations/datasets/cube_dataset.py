import os
import pickle
from scipy.spatial.transform import Rotation
import torch
import roma
from torch.utils.data import Dataset


class CubeImageToPoseDataset(Dataset):
    def __init__(self, args, device, dataset_file, name):
        rots = Rotation.random(args.dataset_size)
        quats = rots.as_quat()

        self.quats = torch.from_numpy(quats)
        self.imgs = []
        dataset_file = dataset_file + "_" + name + ".pkl"

        if os.path.exists(dataset_file):
            dic = pickle.load(open(dataset_file, "rb"))
            self.imgs, self.quats = dic["imgs"], dic["quats"]
            print("Dataset file exists -> loaded")
        else:
            from .dataset_generation import DataGenerator

            dg = DataGenerator(height=args.height, width=args.width)
            for i in range(args.dataset_size):
                # TODO normalize data
                self.imgs.append(torch.from_numpy(dg.render_img(quats[i])))
            dic = {"imgs": self.imgs, "quats": self.quats}
            pickle.dump(dic, open(dataset_file, "wb"))
            print("Dataset file was created and saved")

        self.imgs = [i.to(device) for i in self.imgs]
        self.quats = self.quats.to(device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx].type(torch.float32) / 255, roma.unitquat_to_rotmat(self.quats[idx]).type(torch.float32)


class PoseToCubeImageDataset(CubeImageToPoseDataset):
    def __init__(self, args, device, dataset_file, name):
        super(PoseToCubeImageDataset, self).__init__(args, device, dataset_file, name)

    def __getitem__(self, idx):
        return roma.unitquat_to_rotmat(self.quats[idx]).type(torch.float32), self.imgs[idx].type(torch.float32) / 255

from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):
    def __init__(self, pcd_path, rotated_pcd_path, out_rot_path, representation: Representation):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pcd = np.load(pcd_path)
        self.rotated_pcd = np.load(rotated_pcd_path)
        self.out_rot = np.load(out_rot_path)
        self.ixs = np.arange(len(self.pcd))

        self.f_out = representation.get_f_out()
        self.out_rot, self.ixs = representation.preprocess(self.out_rot, self.ixs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.concatenate((self.pcd[idx], self.rotated_pcd[idx]), axis=-1), device=self.device),
            self.f_out(torch.from_numpy(self.out_rot[idx], device=self.device)),
        )

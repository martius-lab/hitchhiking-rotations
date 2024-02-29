import torch
import numpy as np
import os
from torch.utils.data import Dataset
from hitchhiking_rotations import HITCHHIKING_ROOT_DIR


class PointCloudDataset(Dataset):
    def __init__(self, mode, device):
        path = f"{HITCHHIKING_ROOT_DIR}/assets/datasets/pcd_dataset"
        pcd_path = f"{path}/{mode}_point_cloud.npy"  # N, Npcd, 3
        rotated_pcd_path = f"{path}/rotated_{mode}_point_cloud.npy"
        out_rot_path = f"{path}/{mode}_rotations.npy"

        if not os.path.exists(pcd_path) or not os.path.exists(rotated_pcd_path) or not os.path.exists(out_rot_path):
            print(f"Creating dataset for PCD-ModelNet-Airplane dataset...")
            create_dataset()

        self.device = device

        pcd, rotated_pcd = np.load(pcd_path), np.load(rotated_pcd_path)
        self.feature_input = torch.from_numpy(
            np.concatenate((pcd, rotated_pcd), axis=-1).transpose((0, 2, 1)).astype(np.float32)
        ).to(self.device)
        self.out_rot = torch.from_numpy(np.load(out_rot_path).astype(np.float32)).to(self.device)

        self.N = len(self.feature_input)
        self.ixs = np.arange(self.N)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.feature_input[idx], self.out_rot[idx]


def create_dataset():
    from torchvision import transforms
    import random
    from path import Path
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as R

    def read_off(file):
        if "OFF" != file.readline().strip():
            raise ("Not a valid OFF header")
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(" ")])
        verts = [[float(s) for s in file.readline().strip().split(" ")] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(" ")][1:] for i_face in range(n_faces)]
        return verts, faces

    class PointSampler(object):
        def __init__(self, output_size):
            assert isinstance(output_size, int)
            self.output_size = output_size

        def triangle_area(self, pt1, pt2, pt3):
            side_a = np.linalg.norm(pt1 - pt2)
            side_b = np.linalg.norm(pt2 - pt3)
            side_c = np.linalg.norm(pt3 - pt1)
            s = 0.5 * (side_a + side_b + side_c)
            return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

        def triangel_area_vectorized(self, pt1, pt2, pt3):
            side_a = np.linalg.norm(pt1 - pt2, axis=1)
            side_b = np.linalg.norm(pt2 - pt3, axis=1)
            side_c = np.linalg.norm(pt3 - pt1, axis=1)
            s = 0.5 * (side_a + side_b + side_c)
            return np.fmax(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

        def sample_point_vectorized(self, pt1, pt2, pt3):
            # barycentric coordinates on a triangle
            # https://mathworld.wolfram.com/BarycentricCoordinates.html
            s, t = np.sort(np.random.rand(len(pt1), 2), axis=1).T
            f = lambda i: s * pt1[:, i] + (t - s) * pt2[:, i] + (1 - t) * pt3[:, i]
            return np.stack((f(0), f(1), f(2)), axis=1)

        def sample_point(self, pt1, pt2, pt3):
            # barycentric coordinates on a triangle
            # https://mathworld.wolfram.com/BarycentricCoordinates.html
            s, t = sorted([random.random(), random.random()])
            f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
            return (f(0), f(1), f(2))

        def __call__(self, mesh):
            verts, faces = mesh
            verts = np.array(verts)
            faces = np.array(faces)

            areas = self.triangel_area_vectorized(verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]])

            sampled_faces = np.array(random.choices(faces, weights=areas, cum_weights=None, k=self.output_size))

            sampled_points = self.sample_point_vectorized(
                verts[sampled_faces[:, 0]], verts[sampled_faces[:, 1]], verts[sampled_faces[:, 2]]
            )

            return sampled_points

    class Normalize(object):
        def __call__(self, pointcloud):
            assert len(pointcloud.shape) == 2

            norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
            norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

            return norm_pointcloud

    class PointCloudData(Dataset):
        def __init__(self, root_dir, transform, folder, only_classes=None):
            self.root_dir = root_dir
            folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)]
            self.classes = {folder: i for i, folder in enumerate(folders)}

            if only_classes:
                self.classes = {k: v for k, v in self.classes.items() if k in only_classes}

            self.transforms = transform
            self.files = []
            for category in self.classes.keys():
                new_dir = root_dir / Path(category) / folder
                for file in os.listdir(new_dir):
                    if file.endswith(".off"):
                        sample = {}
                        sample["pcd_path"] = new_dir / file
                        sample["category"] = category
                        self.files.append(sample)

        def __len__(self):
            return len(self.files)

        def __preproc__(self, file):
            verts, faces = read_off(file)
            if self.transforms:
                pointcloud = self.transforms((verts, faces))
            return pointcloud

        def __getitem__(self, idx):
            pcd_path = self.files[idx]["pcd_path"]
            category = self.files[idx]["category"]
            with open(pcd_path, "r") as f:
                pointcloud = self.__preproc__(f)
            return {"pointcloud": pointcloud, "category": self.classes[category]}

    path = Path(f"{HITCHHIKING_ROOT_DIR}/ModelNet40")
    data_path = Path(f"{HITCHHIKING_ROOT_DIR}/assets/datasets/pcd_dataset")

    if not path.exists() and not data_path.exists():
        url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

        if not os.path.exists(path):
            os.makedirs(path)

        # Bash command to download and extract only the "airplane" subfolder
        bash_command = f"wget {url}"
        os.system(bash_command)
        # unzip only the airplane subfolder
        bash_command = f"unzip ModelNet40.zip ModelNet40/airplane/*"
        os.system(bash_command)
        os.remove("ModelNet40.zip")

    # 1. Create train and test point cloud in npy format
    data_path.mkdir_p()
    path_train = data_path / "train_point_cloud.npy"
    path_test = data_path / "test_point_cloud.npy"

    # 1.1 Preprocessing involves sampling 3000 points from the mesh triangle surfaces and normalizing them
    print(f"Sampling and normalizing point clouds...")
    train_transforms = transforms.Compose([PointSampler(3000), Normalize()])

    if not path_train.exists():
        print(f"Creating train point clouds...")
        train_ds = PointCloudData(path, transform=train_transforms, folder="train", only_classes=["airplane"])
        train_pcd = np.array([d["pointcloud"] for d in tqdm(train_ds)]).reshape(-1, 3000, 3)
        # train_pcd2 = np.array([d['pointcloud'] for d in tqdm(train_ds)]).reshape(-1, 3000, 3)
        # train_pcd = np.concatenate((train_pcd, train_pcd2), axis=0)
        np.save(path_train, train_pcd)
    else:
        print(f"File {path_train} already exist")

    if not path_test.exists():
        print(f"Creating test point clouds...")
        test_ds = PointCloudData(path, transform=train_transforms, folder="test", only_classes=["airplane"])
        test_pcd = np.array([d["pointcloud"] for d in tqdm(test_ds)]).reshape(-1, 3000, 3)
        # test_pcd2  = np.array([d['pointcloud'] for d in tqdm(test_ds) ]).reshape(-1, 3000, 3)
        # test_pcd = np.concatenate((test_pcd, test_pcd2), axis=0)
        np.save(path_test, test_pcd)
    else:
        print(f"File {path_test} already exist")

    # 2. Create train and test labels for rotated point clouds

    # Load point clouds
    train_pcd = np.load(path_train)
    test_pcd = np.load(path_test)

    path_train_rotated = data_path / "rotated_train_point_cloud.npy"
    path_train_rots = data_path / "train_rotations.npy"
    path_test_rotated = data_path / "rotated_test_point_cloud.npy"
    path_test_rots = data_path / "test_rotations.npy"

    if not (
        path_train_rotated.exists()
        and path_train_rots.exists()
        and path_test_rotated.exists()
        and path_test_rots.exists()
    ):
        print(f"Creating train and test labels for rotated point clouds...")

        def random_rotates_save(batch_pointcloud, path_rot, path_pcd):
            random_rots = R.random(num=batch_pointcloud.shape[0]).as_matrix()
            rotated_pcd = np.einsum("bkj, bij -> bik", random_rots, batch_pointcloud)
            np.save(path_rot, random_rots)
            np.save(path_pcd, rotated_pcd)
            return random_rots, rotated_pcd

        random_rotates_save(train_pcd, path_train_rots, path_train_rotated)
        random_rotates_save(test_pcd, path_test_rots, path_test_rotated)

    print(f"Dataset creation finished")

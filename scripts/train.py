from scipy.spatial.transform import Rotation
import numpy as np
from torchvision.models import resnet18
from PIL import Image
from torch import nn

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import roma
import numpy as np
import os

from pose_estimation import euler_angles_to_matrix, matrix_to_euler_angles, ToyDataset
from pose_estimation import l2_dp_loss, cosine_similarity_loss, chordal_distance, mse_loss, chordal_loss
from pose_estimation import to_rotmat, get_rotation_representation_dim, to_rotation_representation

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--height", type=int, default=64, help="img_height")
parser.add_argument("--width", type=int, default=64, help="img_width")
parser.add_argument("--dataset_size", type=int, default=2048, help="img_width")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--seed", type=int, default=0, help="number of seeds")
parser.add_argument("--prefix", type=str, default="_chordal", help="number of seeds")
parser.add_argument(
    "--out_dir",
    type=str,
    default="/media/jfrey/git/pose_estimation/DenseFusion/pose_estimation/results/img_to_pose",
    help="batch_size",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/media/jfrey/git/pose_estimation/DenseFusion/pose_estimation/data",
    help="batch_size",
)

args = parser.parse_args()


class Trainer:
    def __init__(self, rotation_representation, device, args, metric="l2"):
        if metric == "l2":
            self.loss = torch.nn.MSELoss()
        elif metric == "dp":
            self.loss = l2_dp
        elif metric == "cosine_similarity":
            self.loss = cosine_similarity
        elif metric == "chordal":
            self.loss = chordal_loss

        self.rotation_representation = rotation_representation
        self.rotation_representation_dim = get_rotation_representation_dim(rotation_representation)

        self.input_dim = int(args.width * args.height * 3)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.rotation_representation_dim),
        )
        # self.model = resnet18(weights=None, num_classes=self.rotation_representation_dim)

        self.model.to(device)
        self.device = device

        # previously 0.01 worked kind of
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.reset()

    def train_batch(self, x, target):
        self.opt.zero_grad()
        pred = self.model(x.reshape(-1, self.input_dim))

        with torch.no_grad():
            target_rep = to_rotation_representation(target, self.rotation_representation)

        loss = self.loss(pred, target_rep, self.rotation_representation)
        loss.backward()
        self.opt.step()

        self.loss_sum_train += loss.item()
        self.count_train += 1

        return loss

    @torch.no_grad()
    def test_batch(self, x, target):
        pred = self.model(x.reshape(-1, self.input_dim))
        with torch.no_grad():
            target_rep = to_rotation_representation(target, self.rotation_representation)

        pred_base = to_rotmat(pred, self.rotation_representation)

        # Alternative you can use: roma.rotmat_geodesic_distance
        self.loss_sum_test += chordal_distance(pred_base, target).mean().item()
        self.count_test += 1

    def reset(self):
        self.loss_sum_train = 0
        self.count_train = 0
        self.loss_sum_test = 0
        self.count_test = 0

    def get_epoch_summary(self, name, verbose):
        tr = self.loss_sum_train / self.count_train
        te = self.loss_sum_test / self.count_test

        if verbose:
            tr_str = str(round(tr, 6))
            te_str = str(round(te, 6))
            print(f"{name}".ljust(15) + f"-- Train loss (mse): {tr_str} -- Test average (chordal_distance) : {te_str}")

        return tr, te


s = args.seed
torch.manual_seed(s)
np.random.seed(s)

device = "cuda"
trainers = {}
# trainers["euler_l2"] = Trainer("euler", device=device, args=args)
# trainers["rotvec_l2"] = Trainer("rotvec", device=device, args=args)

# trainers["quaternion_fixed_l2"] = Trainer("quaternion", device=device, args=args)
# trainers["quaternion_rf_l2"] = Trainer("quaternion_rand_flip", device=device, args=args)
# trainers["quaternion_dp"] = Trainer("quaternion", device=device, args=args, metric="dp")

# trainers["quaternion_canonical_l2"] = Trainer("quaternion_canonical", device=device, args=args)
# trainers["quaternion_canonical_cosine_similarity"] = Trainer(
#     "quaternion_canonical", device=device, args=args, metric="cosine_similarity"
# )
# trainers["gramschmidt_l2"] = Trainer("gramschmidt", device=device, args=args)
# trainers["procrustes_l2"] = Trainer("procrustes", device=device, args=args)


trainers["euler_chordal"] = Trainer("euler", device=device, args=args, metric="chordal")
trainers["rotvec_chordal"] = Trainer("rotvec", device=device, args=args, metric="chordal")
trainers["quaternion_chordal"] = Trainer("quaternion", device=device, args=args, metric="chordal")
trainers["gramschmidt_chordal"] = Trainer("gramschmidt", device=device, args=args, metric="chordal")
trainers["procrustes_chordal"] = Trainer("procrustes", device=device, args=args, metric="chordal")

training_data = ToyDataset(args=args, device=device, dataset_file=args.dataset_file, name="train")
train_dataloader = DataLoader(training_data, num_workers=0, batch_size=args.batch_size, shuffle=True)

test_data = ToyDataset(args=args, device=device, dataset_file=args.dataset_file, name="test")
test_dataloader = DataLoader(test_data, num_workers=0, batch_size=args.batch_size, shuffle=True)

train_losses, test_losses = {n: [] for n in trainers.keys()}, {n: [] for n in trainers.keys()}

for epoch in range(args.epochs):
    for j, batch in enumerate(train_dataloader):
        x, target = batch

        for name, trainer in trainers.items():
            trainer.train_batch(x, target)

    for j, batch in enumerate(test_dataloader):
        x, target = batch

        for name, trainer in trainers.items():
            trainer.test_batch(x, target)

    print(f"Epoch {epoch}:")
    for name, trainer in trainers.items():
        train_loss, test_loss = trainer.get_epoch_summary(name=name, verbose=True)
        train_losses[name].append(train_loss)
        test_losses[name].append(test_loss)
        trainer.reset()
    print("")

for name, trainer in trainers.items():
    print(f"{name} -- Best Train loss (mse on given representation regression): ", np.array(train_losses[name]).min())
    print(f"{name} -- Best Test average (chordal_distance): ", np.array(test_losses[name]).min())

pf = args.prefix
np.save(os.path.join(args.out_dir, f"seed_{s}_train_losses{pf}.npy"), train_losses)
np.save(os.path.join(args.out_dir, f"seed_{s}_test_losses{pf}.npy"), test_losses)

from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle
from hitchhiking_rotations.exp_cfgs import get_cfg_pcd_to_pose, get_cfg_cube_image_to_pose, get_cfg_pose_to_cube_image

import numpy as np
import argparse
import os
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import copy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment",
    type=str,
    choices=["cube_image_to_pose", "pose_to_cube_image", "pose_to_pose"],
    default="cube_image_to_pose",
    help="Experiment Configuration",
)
parser.add_argument("--seed", type=int, default=0, help="number of seeds")
args = parser.parse_args()

s = args.seed
torch.manual_seed(s)
np.random.seed(s)
device = "cuda"

if args.experiment == "cube_image_to_pose":
    cfg_exp = get_cfg_cube_image_to_pose(device)
elif args.experiment == "cube_pose_to_cube_image":
    cfg_exp = get_cfg_pose_to_cube_image(device)
elif args.experiment == "cube_pcd_to_pose":
    cfg_exp = get_cfg_pcd_to_pose(device)

OmegaConf.register_new_resolver("u", lambda x: hydra.utils.get_method("hitchhiking_rotations.utils." + x))
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

cfg_exp = OmegaConf.create(cfg_exp)

trainers = hydra.utils.instantiate(cfg_exp.trainers)
training_data = hydra.utils.instantiate(cfg_exp.training_data)
test_data = hydra.utils.instantiate(cfg_exp.test_data)
val_data = hydra.utils.instantiate(cfg_exp.val_data)

# Create dataloaders
train_dataloader = DataLoader(training_data, num_workers=0, batch_size=cfg_exp.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, num_workers=0, batch_size=cfg_exp.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, num_workers=0, batch_size=cfg_exp.batch_size, shuffle=True)

# Training loop
training_result = {}
for epoch in range(cfg_exp.epochs):
    if cfg_exp.verbose:
        print("\nEpoch: ", epoch)

    # Check if at least one trainer has not stopped based on early stopping
    continue_training = False
    for name, trainer in trainers.items():
        if not trainer.early_stopper.early_stopped:
            continue_training = True
    if not continue_training:
        break

    # Reset logging
    for name, trainer in trainers.items():
        trainer.logger.reset()

    # Perform training
    for j, batch in enumerate(train_dataloader):
        x, target = batch

        for name, trainer in trainers.items():
            if trainer.early_stopper.early_stopped:
                continue

            trainer.train_batch(x.clone(), target.clone(), epoch)

    # Perform validation
    for j, batch in enumerate(val_dataloader):
        x, target = batch

        for name, trainer in trainers.items():
            trainer.test_batch(x.clone(), target.clone(), epoch, mode="val")

    # Store results and update early stopping
    for name, trainer in trainers.items():
        metric = trainer.logger.modes["val"]["geodesic_distance"]
        score = metric["sum"] / metric["count"]

        trainer.early_stopper.early_stop(score)
        training_result[name + f"-epoch_{epoch}"] = copy.deepcopy(trainer.logger.modes)

# Perform testing
for j, batch in enumerate(test_dataloader):
    x, target = batch
    for name, trainer in trainers.items():
        trainer.test_batch(x.clone(), target.clone(), epoch, mode="test")


for name, trainer in trainers.items():
    training_result[name + "-test"] = copy.deepcopy(trainer.logger.modes)

experiment_folder = os.path.join(HITCHHIKING_ROOT_DIR, "results", args.experiment)
os.makedirs(experiment_folder, exist_ok=True)
save_pickle(training_result, os.path.join(experiment_folder, f"seed_{s}_result.npy"))

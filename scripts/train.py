from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle
from hitchhiking_rotations.cfgs import (get_cfg_pcd_to_pose, get_cfg_cube_image_to_pose, get_cfg_pose_to_cube_image,
                                        get_cfg_pose_to_fourier)

import numpy as np
import argparse
import os
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

parser = argparse.ArgumentParser()

fourier_choices = [f"pose_to_fourier_{idx}" for idx in range(1, 7)]

parser.add_argument(
    "--experiment",
    type=str,
    choices=["cube_image_to_pose", "pose_to_cube_image", "pcd_to_pose"] + fourier_choices,
    default="pose_to_cube_image",
    help="Experiment Configuration",
)
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed used during training, " +
                         "for pose_to_fourier the seed is used to select the target function.")
args = parser.parse_args()

s = args.seed
torch.manual_seed(s)
np.random.seed(s)
device = "cuda" if torch.cuda.is_available() else "cpu"
validate_every_n = 5  # This parameter also scales the patience of the early_stopping

if args.experiment == "cube_image_to_pose":
    cfg_exp = get_cfg_cube_image_to_pose(device)

elif args.experiment == "pose_to_cube_image":
    cfg_exp = get_cfg_pose_to_cube_image(device)

elif args.experiment.find("pose_to_fourier") != -1:
    cfg_exp = get_cfg_pose_to_fourier(device, nf=s, nb=int(args.experiment.split("_")[-1]))

elif args.experiment == "pcd_to_pose":
    cfg_exp = get_cfg_pcd_to_pose(device)

OmegaConf.register_new_resolver("u", lambda x: hydra.utils.get_method("hitchhiking_rotations.utils." + x))

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
    for j, batch in enumerate(bar := tqdm(train_dataloader, ncols=100, desc=f"Train-Epoch {epoch}")):
        x, target = batch

        for name, trainer in trainers.items():
            if trainer.early_stopper.early_stopped:
                continue

            trainer.train_batch(x.clone(), target.clone(), epoch)

        if cfg_exp.verbose:
            scores = [t.logger.get_score("train", "loss") for t in trainers.values()]
            bar.set_postfix({"running_train_loss": np.array(scores).mean()})

    if validate_every_n > 0 and epoch % validate_every_n == 0:
        # Perform validation
        for j, batch in enumerate(bar := tqdm(val_dataloader, ncols=100, desc=f"Val-Epoch   {epoch}")):
            x, target = batch

            for name, trainer in trainers.items():
                trainer.test_batch(x.clone(), target.clone(), epoch, mode="val")

        # Store results and update early stopping
        for name, trainer in trainers.items():
            trainer.validation_epoch_finish(epoch)
            training_result[name + f"-epoch_{epoch}"] = copy.deepcopy(trainer.logger.modes)

        for name, trainer in trainers.items():
            training_result[name + f"-epoch_{epoch}"] = copy.deepcopy(trainer.logger.modes)

        if cfg_exp.verbose:
            scores = [t.logger.get_score("val", "loss") for t in trainers.values()]
            bar.set_postfix({"running_val_loss": np.array(scores).mean()})

for name, trainer in trainers.items():
    trainer.training_finish()

# Perform testing
for j, batch in enumerate(tqdm(test_dataloader, ncols=100, desc="Test-Epoch ")):
    x, target = batch
    for name, trainer in trainers.items():
        trainer.test_batch(x.clone(), target.clone(), epoch, mode="test")


for name, trainer in trainers.items():
    training_result[name + "-test"] = copy.deepcopy(trainer.logger.modes)

experiment_folder = os.path.join(HITCHHIKING_ROOT_DIR, "results", args.experiment)
models_folder = os.path.join(HITCHHIKING_ROOT_DIR, "results", args.experiment, "models")
os.makedirs(experiment_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)
save_pickle(training_result, os.path.join(experiment_folder, f"seed_{s}_result.npy"))

for name, trainer in trainers.items():
    p = os.path.join(models_folder, f"seed_{s}_{name}.pt")
    torch.save(trainer.model.state_dict(), p)

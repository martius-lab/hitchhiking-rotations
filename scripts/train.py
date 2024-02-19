from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle

import numpy as np
import argparse
import os
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import hitchhiking_rotations as hr

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


# Current idea:
# 1. dataset provides x, target (being a rotmat)
# 2. apply the preprocess_target with torch.no_grad -> e.g. to transform to quat
# 3. preprocess the input (image) -> e.g. flatten img to correct shape
# 4. inference the network
# 5. apply a postprocessing function before loss -> e.g. Normalization; SVD; GSO
# 6. compute the loss (loss needs to be able to handle everything correctly)
# 7. to the pred after postprocessing apply furhter function before feeding to logging -> e.g. convert quat to rotmat
# 8. currently trainer is responsible for logging and receives the original_target and "post_post_processed_pred"

shared_trainer_cfg = {
    "_target_": "hitchhiking_rotations.utils.Trainer",
    "lr": 0.001,
    "optimizer": "SGD",
    "logger": "${logger}",
    "verbose": True,
    "device": device,
}

cfg_exp = {
    "model9": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 9},
    "model6": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 6},
    "model4": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 4},
    "model3": {"_target_": "hitchhiking_rotations.models.MLP", "input_dim": 12288, "output_dim": 3},
    "logger": {
        "_target_": "hitchhiking_rotations.utils.Logger",
        "metrics": ["l1", "l2", "geodesic_distance", "chordal_distance"],
    },
    "trainers": {
        "r9_l1": {
            **shared_trainer_cfg,
            **{
                "preprocess_input": "${get_method:hitchhiking_rotations.utils.flatten}",
                "preprocess_target": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "postprocess_pred": "${get_method:hitchhiking_rotations.utils.procrustes_to_rotmat}",
                "postprocess_logging": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "loss": "${get_method:hitchhiking_rotations.utils.l2}",
                "model": "${model9}",
            },
        },
        "r9_l2": {
            **shared_trainer_cfg,
            **{
                "preprocess_input": "${get_method:hitchhiking_rotations.utils.flatten}",
                "preprocess_target": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "postprocess_pred": "${get_method:hitchhiking_rotations.utils.procrustes_to_rotmat}",
                "postprocess_logging": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "loss": "${get_method:hitchhiking_rotations.utils.l2}",
                "model": "${model9}",
            },
        },
        "r9_geodesic_distance": {
            **shared_trainer_cfg,
            **{
                "preprocess_input": "${get_method:hitchhiking_rotations.utils.flatten}",
                "preprocess_target": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "postprocess_pred": "${get_method:hitchhiking_rotations.utils.procrustes_to_rotmat}",
                "postprocess_logging": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "loss": "${get_method:hitchhiking_rotations.utils.geodesic_distance}",
                "model": "${model9}",
            },
        },
        "r9_chordal_distance": {
            **shared_trainer_cfg,
            **{
                "preprocess_input": "${get_method:hitchhiking_rotations.utils.flatten}",
                "preprocess_target": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "postprocess_pred": "${get_method:hitchhiking_rotations.utils.procrustes_to_rotmat}",
                "postprocess_logging": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "loss": "${get_method:hitchhiking_rotations.utils.chordal_distance}",
                "model": "${model9}",
            },
        },
        "quat_chordal_distance": {
            **shared_trainer_cfg,
            **{
                "preprocess_input": "${get_method:hitchhiking_rotations.utils.flatten}",
                "preprocess_target": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "postprocess_pred": "${get_method:hitchhiking_rotations.utils.procrustes_to_rotmat}",
                "postprocess_logging": "${get_method:hitchhiking_rotations.utils.passthrough}",
                "loss": "${get_method:hitchhiking_rotations.utils.chordal_distance}",
                "model": "${model4}",
            },
        },
    },
    "batch_size": 32,
    "epochs": 1000,
    "training_data": {
        "_target_": "hitchhiking_rotations.datasets.CubeImageToPoseDataset",
        "mode": "train",
        "dataset_size": 2048,
        "device": device,
    },
    "test_data": {
        "_target_": "hitchhiking_rotations.datasets.CubeImageToPoseDataset",
        "mode": "test",
        "dataset_size": 2048,
        "device": device,
    },
}

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

cfg_exp = OmegaConf.create(cfg_exp)

trainers = hydra.utils.instantiate(cfg_exp.trainers)
training_data = hydra.utils.instantiate(cfg_exp.training_data)
test_data = hydra.utils.instantiate(cfg_exp.test_data)

# Create dataloaders
train_dataloader = DataLoader(training_data, num_workers=0, batch_size=cfg_exp.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, num_workers=0, batch_size=cfg_exp.batch_size, shuffle=True)

# Training loop
for epoch in range(cfg_exp.epochs):
    for j, batch in enumerate(train_dataloader):
        x, target = batch

        for name, trainer in trainers.items():
            trainer.train_batch(x.clone(), target.clone())

    for j, batch in enumerate(test_dataloader):
        x, target = batch

        for name, trainer in trainers.items():
            trainer.test_batch(x.clone(), target.clone())

training_result = {}

experiment_folder = join(HITCHHIKING_ROOT_DIR, "results", args.experiment)
of.makedirs(experiment_folder, exist_ok=True)
save_pickle(join(experiment_folder, f"seed_{s}_result.npy"), training_result)

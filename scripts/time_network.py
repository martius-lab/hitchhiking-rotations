from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle
from hitchhiking_rotations.cfgs import get_cfg_cube_image_to_pose
import numpy as np
import argparse
import os
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed used during training, " + "for pose_to_fourier the seed is used to select the target function.",
)
args = parser.parse_args()

s = args.seed
torch.manual_seed(s)
np.random.seed(s)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg_exp = get_cfg_cube_image_to_pose(device)
OmegaConf.register_new_resolver("u", lambda x: hydra.utils.get_method("hitchhiking_rotations.utils." + x))
cfg_exp = OmegaConf.create(cfg_exp)

trainers = hydra.utils.instantiate(cfg_exp.trainers)
test_data = hydra.utils.instantiate(cfg_exp.test_data)
# Create dataloaders
epoch = 0

timing_result = {}

for batch_size in [256]:  # [1,32,256,1024]:
    test_dataloader = DataLoader(test_data, num_workers=0, batch_size=batch_size, shuffle=True)
    # Perform testing
    for j, batch in enumerate(test_dataloader):
        x, target = batch
        for trainer_name in ["r9_svd_geodesic_distance", "r9_geodesic_distance", "r6_gso_geodesic_distance"]:
            trainer = trainers[trainer_name]

            for i in range(100):
                trainer.test_batch(x.clone(), target.clone(), epoch, mode="test")

            timing_result[f"test_epoch - {batch_size} - {trainer_name}"] = []
            for i in range(100):
                rand = torch.rand_like(x) * 0.00001
                res = trainer.test_batch_time(x + rand, target, epoch, mode="test")
                timing_result[f"test_epoch - {batch_size} - {trainer_name}"].append(res)

        for k in timing_result.keys():
            times, _ = np.array(timing_result[k])
            print(times.shape)
            print(k, " mean t0-t5 ", times.mean(axis=0))
            print(k, " std t0-t5  ", times.std(axis=0))

        break


experiment_folder = os.path.join(HITCHHIKING_ROOT_DIR, "results", "image_to_pose_timing")
os.makedirs(experiment_folder, exist_ok=True)

save_pickle(timing_result, os.path.join(experiment_folder, f"seed_{s}_result.npy"))

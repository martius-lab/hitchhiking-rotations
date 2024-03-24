from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
from hitchhiking_rotations.utils import save_pickle
from hitchhiking_rotations.utils import RotRep
from hitchhiking_rotations.cfgs import get_cfg_cube_image_to_pose
import numpy as np
import argparse
import os
import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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
torch.cuda.empty_cache()
torch.zeros((1,), device=device)  # Initialize CUDA

cfg_exp = get_cfg_cube_image_to_pose(device)
OmegaConf.register_new_resolver("u", lambda x: hydra.utils.get_method("hitchhiking_rotations.utils." + x))
cfg_exp = OmegaConf.create(cfg_exp)

trainers = hydra.utils.instantiate(cfg_exp.trainers)
test_data = hydra.utils.instantiate(cfg_exp.test_data)
# Create dataloaders
epoch = 0

timing_result = {}
batch_sizes = [1, 32, 256, 1024]
for batch_size in batch_sizes:
    test_dataloader = DataLoader(test_data, num_workers=0, batch_size=batch_size, shuffle=True)
    # Perform testing
    for j, batch in enumerate(test_dataloader):
        x, target = batch
        for trainer_name, pretty_name in zip(
            ["r9_svd_geodesic_distance", "r9_geodesic_distance", "r6_gso_geodesic_distance"],
            [str(s) for s in [RotRep.SVD, RotRep.ROTMAT, RotRep.GSO]],
        ):
            trainer = trainers[trainer_name]

            for i in range(100):
                trainer.test_batch(x.clone(), target.clone(), epoch, mode="test")

            timing_result[f"{pretty_name} \n BS-{batch_size}"] = []
            for i in range(100):
                rand = torch.rand_like(x) * 0.00001
                res, _ = trainer.test_batch_time(x + rand, target, epoch, mode="test")
                timing_result[f"{pretty_name} \n BS-{batch_size}"].append(res)

        for k in timing_result.keys():
            times = np.array(timing_result[k])
            print(times.shape)
            print(k, " mean t0-t5 ", times.mean(axis=0))
            print(k, " std t0-t5  ", times.std(axis=0))

        break


# Chat GPT visualization

# Extract timing data for each method
method_names = list(timing_result.keys())
sub_timings = np.array([timing_result[k] for k in method_names])

# Calculate means for each subtiming
means = sub_timings.mean(axis=1)

# Create stacked bar plot
fig, ax = plt.subplots()

bar_width = 0.8

index = []
c = 0
for b in range(len(batch_sizes)):
    for i in range(len(method_names) // len(batch_sizes)):
        index.append(c)
        c += 1
    c += 0.5
index = np.array(index)

plots = []
bottom = np.zeros(len(method_names))

colors = plt.get_cmap("tab20b")([0, 2, 4, 6, 8, 10, 12])


# colors = plt.get_cmap('Greens')(np.linspace(0.0, 1.0, sub_timings.shape[2]))

subtiming_labels = [
    "preprocess_input",
    "model_forward",
    "postprocess_pred_loss",
    "preprocess_target",
    "loss",
    "postprocess_pred_logging",
]

for i, label in enumerate(subtiming_labels):
    plot = ax.bar(index, means[:, i], bar_width, bottom=bottom, label=label, color=colors[i])
    bottom += means[:, i]
    plots.append(plot)

ax.set_xlabel("Methods")
ax.set_ylabel("Time in ms")
ax.set_title("Timing Results")
ax.set_xticks(index)
ax.set_xticklabels(method_names)
ax.legend()
plt.show()


experiment_folder = os.path.join(HITCHHIKING_ROOT_DIR, "results", "image_to_pose_timing")
os.makedirs(experiment_folder, exist_ok=True)

save_pickle(timing_result, os.path.join(experiment_folder, f"seed_{s}_result.npy"))

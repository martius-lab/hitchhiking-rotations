from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
import os
import sys


exps = ["pcd_to_pose", "cube_image_to_pose", "pose_to_cube_image"]
assert len(sys.argv) == 2, "Please provide the experiment name as an argument."
assert sys.argv[1] in exps, f"Experiment name should be one of {exps}"

exp = sys.argv[1]
p = os.path.join(HITCHHIKING_ROOT_DIR, "scripts", "train.py")
for seed in range(10):
    os.system(f"python3 {p} --experiment {exp} --seed {seed}")
if exp in exps[:2]:
    os.system("python3 " + str(os.path.join(HITCHHIKING_ROOT_DIR, "visu", "figure_pointcloud.py ") + exp))
    os.system("python3 " + str(os.path.join(HITCHHIKING_ROOT_DIR, "visu", "figure_exp2a.py ") + exp))
else:
    os.system("python3 " + str(os.path.join(HITCHHIKING_ROOT_DIR, "visu", "figure_exp2b.py")))

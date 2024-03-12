from hitchhiking_rotations import HITCHHIKING_ROOT_DIR
import os

p = os.path.join(HITCHHIKING_ROOT_DIR, "scripts", "train.py")

for seed in range(10):
    os.system(f"python3 {p} --experiment cube_image_to_pose --seed {seed}")

for seed in range(10):
    os.system(f"python3 {p} --experiment pose_to_cube_image --seed {seed}")


os.system("python3 " + str(os.path.join(HITCHHIKING_ROOT_DIR, "visu", "figure_19.py")))
os.system("python3 " + str(os.path.join(HITCHHIKING_ROOT_DIR, "visu", "figure_12a.py")))
os.system("python3 " + str(os.path.join(HITCHHIKING_ROOT_DIR, "visu", "figure_12b.py")))

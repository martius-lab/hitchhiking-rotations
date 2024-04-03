import os
from tqdm import tqdm
from path import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.spatial.transform import Rotation
import seaborn as sns

from hitchhiking_rotations import HITCHHIKING_ROOT_DIR


####################################################
#########  DEFINE AMC_PARSER FUNCTIONALITY #########
####################################################
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D


class Joint:
    def __init__(self, name, direction, length, axis, dof, limits):
        """
        Definition of basic joint. The joint also contains the information of the
        bone between it's parent joint and itself. Refer
        [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
        for detailed description for asf files.

        Parameter
        ---------
        name: Name of the joint defined in the asf file. There should always be one
        root joint. String.

        direction: Default direction of the joint(bone). The motions are all defined
        based on this default pose.

        length: Length of the bone.

        axis: Axis of rotation for the bone.

        dof: Degree of freedom. Specifies the number of motion channels and in what
        order they appear in the AMC file.

        limits: Limits on each of the channels in the dof specification

        """
        self.name = name
        self.length = length
        self.limits = np.zeros([3, 2])
        for lm, nm in zip(limits, dof):
            if nm == "rx":
                self.limits[0] = lm
            elif nm == "ry":
                self.limits[1] = lm
            else:
                self.limits[2] = lm
        self.parent = None
        self.children = []

        # -- spatial transformation information --
        self.direction = np.reshape(direction, [3, 1])  # direction from parent to the
        axis = np.deg2rad(axis)
        self.C = euler2mat(*axis)  # rotation matrix from local to parent  pRl
        self.Cinv = np.linalg.inv(self.C)
        self.coordinate = None  # position in world frame
        self.matrix = None  # rotation in world frame
        self.rot_in_parent = None  # rotation in parent frame

    def set_motion(self, motion):
        if self.name == "root":
            ## Want to set the root to be at the origin
            # self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
            # rotation = np.deg2rad(motion['root'][3:])
            self.coordinate = np.zeros([3, 1])
            rotation = np.zeros(3)

            self.rot_in_parent = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
            self.matrix = self.rot_in_parent
        else:
            idx = 0
            rotation = np.zeros(3)
            for axis, lm in enumerate(self.limits):
                if not np.array_equal(lm, np.zeros(2)):
                    rotation[axis] = motion[self.name][idx]
                    idx += 1
            rotation = np.deg2rad(rotation)  # rotation in local frame  R^l
            #   R^p = pRl * R^l * lRp: (right to left) rot_back_to_parent <- rotate_in_local <- rot_from_parent_to_local
            self.rot_in_parent = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)  # R^p
            self.matrix = self.parent.matrix.dot(self.rot_in_parent)
            self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)

        for child in self.children:
            child.set_motion(motion)

    def draw(self):
        joints = self.to_dict()
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        ax.set_xlim3d(-50, 10)
        ax.set_ylim3d(-20, 40)
        ax.set_zlim3d(-20, 40)

        xs, ys, zs = [], [], []
        for joint in joints.values():
            xs.append(joint.coordinate[0, 0])
            ys.append(joint.coordinate[1, 0])
            zs.append(joint.coordinate[2, 0])
        plt.plot(zs, xs, ys, "b.")

        for joint in joints.values():
            child = joint
            if child.parent is not None:
                parent = child.parent
                xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
                ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
                zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
                plt.plot(zs, xs, ys, "r")
        plt.show()

    def to_dict(self):
        ret = {self.name: self}
        for child in self.children:
            ret.update(child.to_dict())
        return ret

    def pretty_print(self):
        print("===================================")
        print("joint: %s" % self.name)
        print("direction:")
        print(self.direction)
        print("limits:", self.limits)
        print("parent:", self.parent)
        print("children:", self.children)


def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx


def parse_asf(file_path):
    """read joint data only"""
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        # meta infomation is ignored
        if line == ":bonedata":
            content = content[idx + 1 :]
            break

    # read joints
    joints = {"root": Joint("root", np.zeros(3), 0, np.zeros(3), [], [])}
    idx = 0
    while True:
        # the order of each section is hard-coded

        line, idx = read_line(content, idx)

        if line[0] == ":hierarchy":
            break

        assert line[0] == "begin"

        line, idx = read_line(content, idx)
        assert line[0] == "id"

        line, idx = read_line(content, idx)
        assert line[0] == "name"
        name = line[1]

        line, idx = read_line(content, idx)
        assert line[0] == "direction"
        direction = np.array([float(axis) for axis in line[1:]])

        # skip length
        line, idx = read_line(content, idx)
        assert line[0] == "length"
        length = float(line[1])

        line, idx = read_line(content, idx)
        assert line[0] == "axis"
        assert line[4] == "XYZ"

        axis = np.array([float(axis) for axis in line[1:-1]])

        dof = []
        limits = []

        line, idx = read_line(content, idx)
        if line[0] == "dof":
            dof = line[1:]
            for i in range(len(dof)):
                line, idx = read_line(content, idx)
                if i == 0:
                    assert line[0] == "limits"
                    line = line[1:]
                assert len(line) == 2
                mini = float(line[0][1:])
                maxi = float(line[1][:-1])
                limits.append((mini, maxi))

            line, idx = read_line(content, idx)

        assert line[0] == "end"
        joints[name] = Joint(name, direction, length, axis, dof, limits)

    # read hierarchy
    assert line[0] == ":hierarchy"

    line, idx = read_line(content, idx)

    assert line[0] == "begin"

    while True:
        line, idx = read_line(content, idx)
        if line[0] == "end":
            break
        assert len(line) >= 2
        for joint_name in line[1:]:
            joints[line[0]].children.append(joints[joint_name])
        for nm in line[1:]:
            joints[nm].parent = joints[line[0]]

    return joints


def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ":DEGREES":
            content = content[idx + 1 :]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames


####################################################
####################### MAIN #######################
####################################################
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 18})
plt.rcParams.update({"figure.figsize": (7.0, 4.0)})
colors = [
    (0.368, 0.507, 0.71),
    (0.881, 0.611, 0.142),
    (0.923, 0.386, 0.209),
    (0.56, 0.692, 0.195),
    (0.528, 0.471, 0.701),
    (0.772, 0.432, 0.102),
    (0.572, 0.586, 0.0),
]

data_path = f"{HITCHHIKING_ROOT_DIR}/assets/datasets/motion_capture_dataset"
BASE_DIR = Path(f"{data_path}/all_asfamc")
OUT_PATH = Path(f"{data_path}").mkdir_p()
DS_FULL = Path(f"{data_path}/datasets_full.csv")
DS_SAMPLED = Path(f"{data_path}/datasets_sampled_760.csv")
SAMPLED_10K = Path(f"{data_path}/datasets_10k.csv")
T_POSE_PATH = Path(f"{data_path}/motion_T_pose.csv")
NUMPY_DS = Path(f"{data_path}/dataset/")


def download_dataset():
    zip_file = f"{data_path}/allasfamc.zip"

    if not Path(zip_file).exists():
        # download
        print("\n\nDownloading dataset")
        url = "wget http://mocap.cs.cmu.edu/allasfamc.zip"
        bash_command = f"wget {url} -P {data_path}"
        os.system(bash_command)

    # extract only airplane
    if not BASE_DIR.exists():
        print("Extracting dataset")
        bash_command = f"unzip {zip_file} -d {data_path}"
        os.system(bash_command)


def load():
    def get_joint_pos_rot_arr(joints, amc_path, frame, joint_names):
        c_joints = joints
        c_motion = parse_amc(amc_path)[frame]
        c_joints["root"].set_motion(c_motion)
        d = c_joints["root"].to_dict()

        out_arr = np.array(
            [d[j].coordinate.flatten().tolist() + d[j].rot_in_parent.flatten().tolist() for j in joint_names]
        )
        return out_arr.flatten().tolist()

    if not BASE_DIR.exists():
        download_dataset()

    # All motion clips
    if not DS_FULL.exists():
        print("Creating full dataset")

        def get_nframe(amc_path):
            with open(amc_path) as f:
                content = f.read().splitlines()
            for line in content[::-1]:
                if line[0].isnumeric():
                    return int(line)

        datasets_df = pd.DataFrame({"path": list(BASE_DIR.glob("subjects/*/*.amc"))})
        datasets_df["Subject"] = datasets_df["path"].map(lambda x: x.parent.stem)
        datasets_df["Activity"] = datasets_df["path"].map(lambda x: x.stem.split("_")[-1].lower())
        datasets_df["asf_path"] = datasets_df["path"].map(lambda x: x.parent / (x.parent.stem + ".asf"))
        datasets_df["n_frames"] = datasets_df["path"].map(get_nframe)
        datasets_df.to_csv(DS_FULL, index=False)

    if not DS_SAMPLED.exists():
        print("Creating sampled dataset")
        datasets_df = pd.read_csv(DS_FULL)
        datasets_df = datasets_df[datasets_df["n_frames"] > 100]  # remove motion clips with less than 100 frames

        # Sample 760(pepe) or 865(zhou)
        # sample 30 frames from each motion clip
        print("Creating 760 motions sampled dataset")
        N_motion_samples = 760
        N_frames_sampled = 30
        datasets_s = datasets_df.sample(N_motion_samples, random_state=0).reset_index(drop=True)
        datasets_s["sampled_frames"] = datasets_s["n_frames"].map(
            lambda n_frame: random.sample(range(n_frame), N_frames_sampled)
        )
        datasets_s.to_csv(DS_SAMPLED, index=False)

    # create dataset with 10k entries
    if not SAMPLED_10K.exists():
        print("Creating 10k motions sampled dataset")
        datasets_s = pd.read_csv(DS_SAMPLED, converters={"sampled_frames": pd.eval})

        N_SAMPLES = 10000

        N_motion_samples = len(datasets_s)
        N_frames_sampled = len(datasets_s["sampled_frames"][0])
        N_total_sampled_frames = N_motion_samples * N_frames_sampled

        # Sample 10k frames
        print("Sampling 10k frames dataset")
        ix_sampled = random.sample(range(N_total_sampled_frames), N_SAMPLES)

        # Get the corresponding metadata
        motion_sampled = [ix // N_frames_sampled for ix in ix_sampled]
        frame_sampled = [ix % N_frames_sampled for ix in ix_sampled]
        paths = datasets_s["path"].to_numpy()[motion_sampled]
        asf_paths = datasets_s["asf_path"].to_numpy()[motion_sampled]
        frames = [datasets_s["sampled_frames"][m][f] for m, f in zip(motion_sampled, frame_sampled)]
        subjects = [datasets_s["Subject"][m] for m in motion_sampled]
        activities = [datasets_s["Activity"][m] for m in motion_sampled]

        # Save to csv
        dataset_10k = pd.DataFrame(
            {"path": paths, "frame": frames, "Subject": subjects, "Activity": activities, "asf_path": asf_paths}
        )
        joint_names = list(parse_asf(dataset_10k["asf_path"][0]).keys())
        dic_of_joints = {
            asf: parse_asf(asf) for asf in tqdm(dataset_10k["asf_path"].unique(), desc="Parsing asf files")
        }

        tqdm.pandas(desc="Computing joint pos and rot for each motion")
        dataset_10k["pos_rot"] = dataset_10k.progress_apply(
            lambda row: get_joint_pos_rot_arr(dic_of_joints[row.asf_path], row.path, row.frame, joint_names), axis=1
        )

        print("Saving dataset")
        dataset_10k.to_csv(SAMPLED_10K, index=False)

    # 1. Capture first frame in T-Pose (87_02)
    if not T_POSE_PATH.exists():
        print("Creating T-Pose")
        subject, activity = "87", "02"
        tpose_amc_path = BASE_DIR / "subjects" / subject / f"{subject}_{activity}.amc"
        tpose_asf_path = BASE_DIR / "subjects" / subject / f"{subject}.asf"
        assert tpose_amc_path.exists(), "T-Pose amc path does not exist"
        assert tpose_asf_path.exists(), "T-Pose asf path does not exist"

        joints = parse_asf(tpose_asf_path)  # dict of joints: forward kinematics

        joint_names = list(joints.keys())

        t_pose_pos_rot = get_joint_pos_rot_arr(joints, tpose_amc_path, frame=0, joint_names=joint_names)

        t_pose_df = pd.DataFrame(
            {
                "Subject": ["87"],
                "Activity": ["02"],
                "path": [tpose_amc_path],
                "asf_path": [tpose_asf_path],
                "pos_rot": [t_pose_pos_rot],
            }
        )
        t_pose_df.to_csv(T_POSE_PATH, index=False)

    # load T_pose
    if not NUMPY_DS.exists():
        print("Saving samped dataset w.r.t T-Pose as numpy")
        NUMPY_DS.mkdir_p()
        tpose_df = pd.read_csv(T_POSE_PATH, converters={"pos_rot": pd.eval})
        dataset_df = pd.read_csv(SAMPLED_10K, converters={"pos_rot": pd.eval})

        N_size_dataset = len(dataset_df)
        N_joints = len(tpose_df["pos_rot"][0]) // 12

        t_pose_pos_rot = np.array(tpose_df["pos_rot"][0]).reshape(-1, 12)
        dataset_pos_rot = np.stack([np.array(pos_rot).reshape(-1, 12) for pos_rot in dataset_df["pos_rot"]])

        # T-pose
        t_pose_pos = t_pose_pos_rot[:, :3].reshape((1, N_joints, 3, 1))
        t_pose_rot = t_pose_pos_rot[:, 3:].reshape((1, N_joints, 3, 3))

        # Dataset
        dataset_pos = dataset_pos_rot[:, :, :3, None]
        dataset_rots = dataset_pos_rot[:, :, 3:].reshape((N_size_dataset, N_joints, 3, 3))
        dataset_rot_tpose_2_rot = np.matmul(np.linalg.inv(t_pose_rot), dataset_rots)

        tpose_df.to_csv(NUMPY_DS / "tpose.csv", index=False)
        dataset_df.to_csv(NUMPY_DS / "dataset.csv", index=False)
        np.save(NUMPY_DS / "tpose_pos.npy", t_pose_pos)
        np.save(NUMPY_DS / "tpose_rot.npy", t_pose_rot)
        np.save(NUMPY_DS / "ds_pos.npy", dataset_pos)
        np.save(NUMPY_DS / "ds_rot_tpose_2_rot.npy", dataset_rot_tpose_2_rot)


def plot_distances():
    ds_rot_tpose_2_rot = Rotation.from_matrix(
        np.load(NUMPY_DS / "ds_rot_tpose_2_rot.npy").reshape(-1, 3, 3)
    )  # N*Njoints, 3, 3  = 10k*31, 3, 3

    so3_dist = Rotation.magnitude(ds_rot_tpose_2_rot)
    so3_d_mean = so3_dist.mean()
    so3_d_std = so3_dist.std()

    quat_dist = np.linalg.norm(ds_rot_tpose_2_rot.as_quat(canonical=True) - np.array([[0, 0, 0, 1]]), axis=-1)
    quat_d_mean = quat_dist.mean()
    quat_d_std = quat_dist.std()

    euler_dist = np.linalg.norm(ds_rot_tpose_2_rot.as_euler(seq="xyz"), axis=-1)
    euler_d_mean = euler_dist.mean()
    euler_d_std = euler_dist.std()

    exp_dist = np.linalg.norm(ds_rot_tpose_2_rot.as_rotvec(), axis=-1)
    exp_d_mean = exp_dist.mean()
    exp_d_std = exp_dist.std()

    plt.hist(so3_dist, bins=100, label="SO3", fc=colors[0] + (0.5,))
    plt.hist(quat_dist, bins=100, label="Quaternions", fc=colors[1] + (0.5,))
    plt.axvline(3.14, color=colors[0], linestyle="dashed")
    plt.axvline(np.sqrt(2), color=colors[1], linestyle="dashed")
    plt.locator_params(axis="y", nbins=3)

    # log y axis
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.tight_layout()
    out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", f"mocap_analsysis.pdf")
    plt.savefig(out_p, bbox_inches="tight")
    plt.show()


def plot_skeleton():
    out_p = os.path.join(HITCHHIKING_ROOT_DIR, "results", f"mocap_sceleton.pdf")
    dataset_10k = pd.read_csv(DS_SAMPLED, converters={"sampled_frames": pd.eval})

    asf_path, amc_path, sampled_frames = dataset_10k[["asf_path", "path", "sampled_frames"]].iloc[33]

    c_joints = parse_asf(asf_path)
    c_motion = parse_amc(amc_path)[sampled_frames[2]]
    c_joints["root"].set_motion(c_motion)

    def draw(joint):
        joints = joint.to_dict()
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        ax.set_xlim3d(-30, 30)
        ax.set_ylim3d(-20, 20)
        ax.set_zlim3d(-20, 20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

        for joint in joints.values():
            child = joint
            if child.parent is not None:
                parent = child.parent
                xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
                ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
                zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
                plt.plot(zs, xs, ys, "k", linewidth=0.7)

        xs, ys, zs = [], [], []
        for joint in joints.values():
            xs.append(joint.coordinate[0, 0])
            ys.append(joint.coordinate[1, 0])
            zs.append(joint.coordinate[2, 0])
        plt.plot(zs, xs, ys, "b.")

        # set looking angle 33 , -5 0
        ax.view_init(33, -5)
        plt.savefig(out_p, bbox_inches="tight")
        plt.show()

    draw(c_joints["root"])


if __name__ == "__main__":
    if not NUMPY_DS.exists():
        load()

    plot_distances()

    plot_skeleton()

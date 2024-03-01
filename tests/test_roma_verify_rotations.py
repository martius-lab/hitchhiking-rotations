import torch
import roma
import numpy as np
from scipy.spatial.transform import Rotation as R
from hitchhiking_rotations.utils import euler_angles_to_matrix


def test_roma_quaternion():
    BS = 1
    NR_SAMPLES = 1000
    for i in range(NR_SAMPLES):
        test_rotation = R.random(BS)
        quat_wxyz = torch.from_numpy(test_rotation.as_quat())
        # xyzw
        out = roma.unitquat_to_rotmat(quat_wxyz)

        if np.abs((test_rotation.as_matrix() - out.numpy())).sum() > 0.00001:
            print("Something went wrong.")
            # raise ValueError("Something went wrong.")
    print("test_roma_quaternion - successfully working")


def test_special_gramschmidt():
    BS = 2
    NR_SAMPLES = 1000
    for i in range(NR_SAMPLES):
        test_rotation = R.random(BS)
        mat = torch.from_numpy(test_rotation.as_matrix())
        out_mat = roma.special_gramschmidt(mat[:, :, :2])

        error = (mat - out_mat).sum()
        if error > 0.000001:
            raise ValueError("Something went wrong.")
    print("test_ortho6d - successfully working")


def test_special_procrustes():
    BS = 2
    NR_SAMPLES = 1000
    for i in range(NR_SAMPLES):
        test_rotation = R.random(BS)
        mat = torch.from_numpy(test_rotation.as_matrix())

        out_mat = roma.special_procrustes(mat[:, :, :])

        error = (mat - out_mat).sum()
        if error > 0.000001:
            raise ValueError("Something went wrong.")
    print("test_special_procrustes - successfully working")


def test_euler_pytorch_3d():
    BS = 1
    NR_SAMPLES = 1000
    for i in range(NR_SAMPLES):
        test_rotation = R.random(BS)
        mat = torch.from_numpy(test_rotation.as_matrix())
        euler_extrinsic = torch.from_numpy(test_rotation.as_euler("XZY", degrees=False).astype(np.float32))

        # Sanity check
        R.from_euler("xzy", test_rotation.as_euler("xzy", degrees=False), degrees=False).as_matrix()
        test_rotation.as_matrix()

        out_mat = euler_angles_to_matrix(euler_extrinsic, convention="XZY")
        error = np.abs(mat - out_mat).sum()
        if error > 0.0001:
            raise ValueError("Something went wrong. Intrinsic")

    print("test_euler - successfully working")


if __name__ == "__main__":
    test_roma_quaternion()
    test_special_gramschmidt()
    test_special_procrustes()
    test_euler_pytorch_3d()

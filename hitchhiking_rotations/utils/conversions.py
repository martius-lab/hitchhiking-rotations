#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from hitchhiking_rotations.utils.euler_helper import euler_angles_to_matrix, matrix_to_euler_angles
import roma
import torch
from math import pi


def euler_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return euler_angles_to_matrix(inp.reshape(-1, 3), convention="XZY")


def quaternion_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    # without normalization
    # normalize first
    x = inp.reshape(-1, 4)
    return roma.unitquat_to_rotmat(x / x.norm(dim=1, keepdim=True))


def gramschmidt_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.special_gramschmidt(inp.reshape(-1, 3, 2))


def symmetric_orthogonalization(x, **kwargs):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def procrustes_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return symmetric_orthogonalization(inp)
    return roma.special_procrustes(inp.reshape(-1, 3, 3))


def rotvec_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.rotvec_to_rotmat(inp.reshape(-1, 3))


# rotmat to x / maybe here reshape is missing


def rotmat_to_euler(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return matrix_to_euler_angles(base, convention="XZY")


def rotmat_to_quaternion(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.rotmat_to_unitquat(base)


def rotmat_to_quaternion_rand_flip(base: torch.Tensor, **kwargs) -> torch.Tensor:
    rep = roma.rotmat_to_unitquat(base)
    rand_flipping = torch.rand(base.shape[0]) > 0.5
    rep[rand_flipping] *= -1
    return rep


def rotmat_to_quaternion_canonical(base: torch.Tensor, **kwargs) -> torch.Tensor:
    rep = roma.rotmat_to_unitquat(base)
    rep[rep[:, 3] < 0] *= -1
    return rep


def rotmat_to_quaternion_aug(base: torch.Tensor, mode: str) -> torch.Tensor:
    """Performs memory-efficient quaternion augmentation by randomly
    selecting half of the quaternions in the batch with scalar part
    smaller than 0.1 and then multiplies them by -1.
    """
    rep = rotmat_to_quaternion_canonical(base)

    if mode == "train":
        rep[torch.logical_and(torch.rand(rep.size(0), device=rep.device) < 0.5, rep[:, 3] < 0.3)] *= -1

    return rep


def rotmat_to_gramschmidt(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return base[:, :, :2]


def rotmat_to_gramschmidt_f(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return base[:, :, :2].reshape(-1, 6)


def rotmat_to_procrustes(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return base


def rotmat_to_rotvec(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.rotmat_to_rotvec(base)


def rotmat_to_rotvec_canonical(base: torch.Tensor, **kwargs) -> torch.Tensor:
    """WARNING: THIS FUNCTION HAS NOT BEEN TESTED"""
    rep = roma.rotmat_to_rotvec(base)
    rep[rep[:, 2] < 0] = (1.0 - 2.0 * pi / rep[rep[:, 2] < 0].norm(dim=1, keepdim=True)) * rep[rep[:, 2] < 0]
    return rep


def test_all():
    from scipy.spatial.transform import Rotation
    import numpy as np
    from torch import from_numpy as tr

    rs = Rotation.random(1000)
    euler = rs.as_euler("XZY", degrees=False)
    rot = rs.as_matrix()
    quat = rs.as_quat()
    quat_hm = np.where(quat[:, 3:4] < 0, -quat, quat)
    rotvec = rs.as_rotvec()

    # euler_to_rotmat
    print(np.allclose(euler_to_rotmat(tr(euler)).numpy(), rot))
    print(np.allclose(quaternion_to_rotmat(tr(quat)).numpy(), rot))
    print(np.allclose(rotvec_to_rotmat(tr(rotvec)).numpy(), rot))
    print(np.allclose(gramschmidt_to_rotmat(tr(rot[:, :, :2])).numpy(), rot))
    print(np.allclose(procrustes_to_rotmat(tr(rot)).numpy(), rot))

    print(np.allclose(rotmat_to_euler(tr(rot)).numpy(), euler))
    print(np.allclose(rotmat_to_gramschmidt(tr(rot)).numpy(), rot[:, :, :2]))
    print(np.allclose(rotmat_to_procrustes(tr(rot)).numpy(), rot))
    print(np.allclose(rotmat_to_rotvec(tr(rot)).numpy(), rotvec))

    print(np.allclose(rotmat_to_quaternion_canonical(tr(rot)).numpy(), quat_hm))
    print(np.allclose(np.abs(rotmat_to_quaternion(tr(rot)).numpy()), np.abs(quat)))

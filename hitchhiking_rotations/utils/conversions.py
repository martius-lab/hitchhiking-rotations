from .euler_helper import euler_angles_to_matrix, matrix_to_euler_angles
import roma
import torch

# x to rotmat


def euler_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    return euler_angles_to_matrix(inp.reshape(-1, 3), convention="XZY")


def quaternion_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    # without normalization
    return roma.unitquat_to_rotmat(inp.reshape(-1, 4))


def gramschmidt_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    return roma.special_gramschmidt(inp.reshape(-1, 3, 2))


def procrustes_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    return roma.special_procrustes(inp.reshape(-1, 3, 3))


def rotvec_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    return roma.rotvec_to_rotmat(inp.reshape(-1, 3))


# rotmat to x / maybe here reshape is missing


def rotmat_to_euler(base: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler_angles(base, convention="XZY")


def rotmat_to_quaternion(base: torch.Tensor) -> torch.Tensor:
    return roma.rotmat_to_unitquat(base)


def rotmat_to_quaternion_rand_flip(base: torch.Tensor) -> torch.Tensor:
    rep = roma.rotmat_to_unitquat(base)
    rand_flipping = torch.rand(base.shape[0]) > 0.5
    rep[rand_flipping] *= -1
    return rep


def rotmat_to_quaternion_canonical(base: torch.Tensor) -> torch.Tensor:
    rep = roma.rotmat_to_unitquat(base)
    rep[rep[:, 3] < 0] *= -1
    return rep


def rotmat_to_gramschmidt(base: torch.Tensor) -> torch.Tensor:
    return base[:, :, :2]


def rotmat_to_procrustes(base: torch.Tensor) -> torch.Tensor:
    return base


def rotmat_to_rotvec(base: torch.Tensor) -> torch.Tensor:
    return roma.rotmat_to_rotvec(base)

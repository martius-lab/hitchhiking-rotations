from pose_estimation import euler_angles_to_matrix, matrix_to_euler_angles
import roma
import torch


def get_rotation_representation_dim(rotation_representation: str) -> int:
    """
    Return dimensionality of rotation representation

    Args:
        rotation_representation (str): rotation representation identifier

    Returns:
        int: dimensionality of rotation representation
    """
    if rotation_representation == "euler":
        rotation_representation_dim = 3
    elif rotation_representation == "rotvec":
        rotation_representation_dim = 3
    elif (
        rotation_representation == "quaternion"
        or rotation_representation == "quaternion_canonical"
        or rotation_representation == "quaternion_rand_flip"
    ):
        rotation_representation_dim = 4

    elif rotation_representation == "procrustes":
        rotation_representation_dim = 9
    elif rotation_representation == "gramschmidt":
        rotation_representation_dim = 6
    else:
        raise ValueError("Unknown rotation representation" + rotation_representation)

    return rotation_representation_dim


def to_rotmat(inp: torch.Tensor, rotation_representation: str) -> torch.Tensor:
    """
    Supported representations and shapes:

    quaternion:             N,4     - comment: XYZW
    quaternion_canonical:   N,4     - comment: XYZW
    gramschmidt:            N,3,2   -
    procrustes:             N,3,3   -
    rotvec:                 N,3     -

    Args:
        inp (torch.tensor, shape=(N,..), dtype=torch.float32): specified rotation representation
        rotation_representation (string): rotation representation identifier

    Returns:
        (torch.tensor, shape=(N,...): SO3 Rotation Matrix
    """

    if rotation_representation == "euler":
        base = euler_angles_to_matrix(inp.reshape(-1, 3), convention="XZY")

    elif (
        rotation_representation == "quaternion"
        or rotation_representation == "quaternion_canonical"
        or rotation_representation == "quaternion_rand_flip"
    ):
        inp = inp.reshape(-1, 4)
        # normalize
        inp = inp / torch.norm(inp, dim=1, keepdim=True)
        base = roma.unitquat_to_rotmat(inp.reshape(-1, 4))

    elif rotation_representation == "gramschmidt":
        base = roma.special_gramschmidt(inp.reshape(-1, 3, 2))

    elif rotation_representation == "procrustes":
        base = roma.special_procrustes(inp.reshape(-1, 3, 3))

    elif rotation_representation == "rotvec":
        base = roma.rotvec_to_rotmat(inp.reshape(-1, 3))

    return base


def to_rotation_representation(base: torch.Tensor, rotation_representation: str) -> torch.Tensor:
    """
    Quaternion representation is always XYZW
    For Euler uses XZY

    Args:
        base (torch.tensor, shape=(N,3,3), dtype=torch.float32): SO3 Rotation Matrix
        rotation_representation (string): rotation representation identifier

    Returns:
        (torch.tensor, shape=(N,...): Returns selected rotation representation
    """

    rotation_representation_dim = get_rotation_representation_dim(rotation_representation)
    if rotation_representation == "euler":
        rep = matrix_to_euler_angles(base, convention="XZY")

    elif rotation_representation == "quaternion":
        rep = roma.rotmat_to_unitquat(base)

    elif rotation_representation == "quaternion_rand_flip":
        rep = roma.rotmat_to_unitquat(base)
        rand_flipping = torch.rand(base.shape[0]) > 0.5
        rep[rand_flipping] *= -1

    elif rotation_representation == "quaternion_canonical":
        rep = roma.rotmat_to_unitquat(base)
        rep[rep[:, 3] < 0] *= -1

    elif rotation_representation == "gramschmidt":
        rep = base[:, :, :2]

    elif rotation_representation == "procrustes":
        rep = base

    elif rotation_representation == "rotvec":
        rep = roma.rotmat_to_rotvec(base)

    return rep.reshape(-1, rotation_representation_dim)

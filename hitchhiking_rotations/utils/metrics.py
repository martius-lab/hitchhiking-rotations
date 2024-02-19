import torch
from .conversions import to_rotmat


def chordal_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target, p="fro", dim=[1, 2])


def l2_dp_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Returns distance picking l2 norm

    Args:
        pred (torch.Tensor, shape=(N,4)): Prediction Quaternion XYZW
        target (torch.Tensor, shape=(N,4)): Target Quaternion XYZW

    Returns:
        (torch.Tensor, shape=(N)): distance
    """
    assert pred.shape[1] == 4
    assert target.shape[1] == 4

    with torch.no_grad():
        target_flipped = target.clone()
        target_flipped[target_flipped[:, 3] < 0] *= -1

    normal = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=1)
    flipped = torch.nn.functional.mse_loss(pred, target_flipped, reduction="none").mean(dim=1)

    m1 = normal < flipped
    return (normal[m1].sum() + flipped[~m1].sum()) / m1.numel()


def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(pred, target).mean()


def chordal_loss(pred: torch.Tensor, target: torch.Tensor, rotation_representation: str, **kwargs) -> torch.Tensor:
    base_pred = to_rotmat(pred, rotation_representation)

    with torch.no_grad():
        base_target = to_rotmat(target, rotation_representation)

    return chordal_distance(base_pred, base_target).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target).mean()

import torch
import roma


def chordal_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target).mean()


def cosine_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1 - torch.nn.functional.cosine_similarity(pred, target).mean()


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred / pred.norm(dim=1, keepdim=True)
    return torch.nn.functional.cosine_similarity(pred, target).mean()


def geodesic_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # return roma.rotmat_geodesic_distance_naive(pred, target).mean()
    return roma.rotmat_geodesic_distance(pred, target).mean()


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # return torch.norm(pred - target, p=1, dim=[1, 2]).mean()
    return torch.nn.functional.l1_loss(pred, target).mean()


def l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target).mean()


def l2_dp(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
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


def test_all():
    # Test geodesic distance (make sure it is the same as scipy)
    from scipy.spatial.transform import Rotation

    r1 = Rotation.random()
    r2 = Rotation.random()
    r1_torch = torch.tensor(r1.as_matrix())
    r2_torch = torch.tensor(r2.as_matrix())

    scipy_err = Rotation.magnitude(r1 * r2.inv())
    roma_err = roma.rotmat_geodesic_distance(r1_torch, r2_torch)

    print(scipy_err, roma_err.item())

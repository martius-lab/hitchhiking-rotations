from pytictac import Timer
import torch


def test_svd():
    BS = 1024
    repeats = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(repeats):
        m = torch.rand((BS, 3, 3), device=device)
        u, s, v = torch.svd(m)

    with Timer("SVD"):
        for i in range(repeats):
            m = torch.rand((BS, 3, 3), device=device)
            u, s, v = torch.svd(m)

    m = torch.rand((BS, 3, 3), device=device)
    with Timer("SVD"):
        for i in range(repeats):
            u, s, v = torch.svd(m)

    m = torch.rand((BS, 3, 3), device=device)
    with Timer("SVD single"):
        u, s, v = torch.svd(m)


if __name__ == "__main__":
    test_svd()

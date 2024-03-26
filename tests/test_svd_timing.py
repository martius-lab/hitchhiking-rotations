from pytictac import Timer, CpuTimer
import torch


def test_svd():
    BS = 128
    repeats = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        tim = Timer
    else:
        tim = CpuTimer

    for i in range(repeats):
        m = torch.rand((BS, 3, 3), device=device)
        u, s, v = torch.svd(m)

    with tim("SVD"):
        for i in range(repeats):
            m = torch.rand((BS, 3, 3), device=device)
            u, s, v = torch.svd(m)

    m = torch.rand((BS, 3, 3), device=device)
    with tim("SVD"):
        for i in range(repeats):
            u, s, v = torch.svd(m)

    m = torch.rand((BS, 3, 3), device=device)
    with tim("SVD single"):
        u, s, v = torch.svd(m)


if __name__ == "__main__":
    test_svd()

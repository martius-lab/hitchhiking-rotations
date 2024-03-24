from pytictac import Timer
import torch

BS = 1024
repeats = 100

for i in range(repeats):
    m = torch.rand((BS, 3, 3), device="cuda")
    u, s, v = torch.svd(m)

with Timer("SVD"):
    for i in range(repeats):
        m = torch.rand((BS, 3, 3), device="cuda")
        u, s, v = torch.svd(m)

m = torch.rand((BS, 3, 3), device="cuda")
with Timer("SVD"):
    for i in range(repeats):
        u, s, v = torch.svd(m)


m = torch.rand((BS, 3, 3), device="cuda")
with Timer("SVD single"):
    u, s, v = torch.svd(m)

import torch

import os

device = torch.device("cuda")

a = torch.randn((10000, 10000)).to(device)

while True:
    b = torch.randn((10000, 10000)).to(device)
    c = torch.matmul(a, b)
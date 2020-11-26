import torch

a=torch.rand(size=(2,3))
print(a)
print(a.max(dim=1).values.dtype)
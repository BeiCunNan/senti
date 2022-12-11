import torch
from torch import nn

a = torch.Tensor([[1, 2], [3, 4]])
a=nn.Softmax(dim=-1)(a)
print(a)

import torch
import torch.nn.functional as F

x = torch.randn((2, 3, 4))
print(x.shape[1])
padded_x = F.pad(x.permute(0, 2, 1), (0,5-x.shape[1]), mode='constant', value=0).permute(0, 2, 1)
print(padded_x)

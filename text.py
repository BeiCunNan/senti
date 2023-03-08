import torch

def f(x):
    return torch.add(torch.exp(x) / (torch.exp(x) + 1), 0.5)

# 创建一个大小为(2, 3)的Tensor
x = torch.Tensor(torch.randn([32,52,768]))

print(type(x))

# 计算每个元素在f(x)函数中的值
result = f(x)

print(result)

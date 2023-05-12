import torch
import numpy as np

# tensorについて理解する

data = np.arange(4).reshape((2, 2)) + 1
print(data)

print(type(data))

data = torch.from_numpy(data)
print(data)

print(type(data))
data = data * data

print(data)

# randomなもの

rand_tensor = torch.rand_like(data, dtype=torch.float)
print(rand_tensor)

print(rand_tensor.shape)
print(rand_tensor.dtype)
print(rand_tensor.device)

print(torch.cuda.is_available())

x = data.to(torch.float) @ rand_tensor
print(x)

x = torch.arange(16).reshape((4, 4))
print(x)

print(x[0])
print(x[:, 1])
print(x[..., -1]+3)

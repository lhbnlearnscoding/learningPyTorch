import torch

some_tensor = torch.rand([1, 3, 4])
print(some_tensor)

print(f"Tensor size: {some_tensor.size()}")
print(f"Tensor shape: {some_tensor.shape}")
print(f"Tensor ndim: {some_tensor.ndim}")
print(f"Tensor dtype: {some_tensor.dtype}")
print(f"Tensor device: {some_tensor.device}")

print("First element:", some_tensor[0, 0, 0].item())

zeros_tensor = torch.zeros([1, 3, 4])

print(some_tensor*zeros_tensor)




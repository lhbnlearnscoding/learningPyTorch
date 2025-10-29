import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
tensor_squared = tensor ** 2
print("Original Tensor:\n", tensor)
print("Squared Tensor:\n", tensor_squared)

print("the tensor  mal 10:", tensor * 10)
print("the tensor add 5:", tensor + 5)
print("the tensor sub 3:", tensor - 3)
print("the tensor div 2:", tensor / 2)
print("the tensor pow 3:", tensor ** 3)
print("the tensor sqrt:", torch.sqrt(tensor))

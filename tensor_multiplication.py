import torch

tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
tensor_c = torch.matmul(tensor_a, tensor_b) # faster than using normal multiplication for 2D tensors
print("Tensor A:\n", tensor_a)
print("Tensor B:\n", tensor_b)

print("Matrix Multiplication Result (A @ B):\n", tensor_c)
print("Element-wise Multiplication Result (A * B):\n", tensor_a * tensor_b)
print("Element-wise Addition Result (A + B):\n", tensor_a + tensor_b)
print("Element-wise Subtraction Result (A - B):\n", tensor_a - tensor_b)
print("Element-wise Division Result (A / B):\n", tensor_a / tensor_b)


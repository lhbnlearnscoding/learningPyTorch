import torch

tensor = torch.arange(0,100,10, dtype=torch.float32)
print("Original Tensor:\n", tensor)

print(f"Maximum Value: {torch.max(tensor).item()}")
print(f"Minimum Value: {torch.min(tensor).item()}")
print(f"Sum of Elements: {torch.sum(tensor).item()}")
print(f"Mean Value: {torch.mean(tensor).item()}")
print(f"Standard Deviation: {torch.std(tensor).item()}")
print(f"Variance: {torch.var(tensor).item()}")
print(f"Location of Maximum Values: {torch.argmax(tensor)}")
print(f"Location of Minimum Values: {torch.argmin(tensor)}")

# Reshaping the tensor- reshaoes an input tensor to the specified shape
reshaped_tensor = tensor.reshape(2,5)
print("Reshaped Tensor (2x5):\n", reshaped_tensor)

# View - returns a view of an input tensor of certain shape but keep the same memory as the original tensor
viewed_tensor = tensor.view(5,2)
print("Viewed Tensor (5x2):\n", viewed_tensor)

# Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
tensor_a = torch.tensor([1,2,3])    
tensor_b = torch.tensor([4,5,6])
vstacked_tensor = torch.vstack((tensor_a, tensor_b))
hstacked_tensor = torch.hstack((tensor_a, tensor_b))
print("Vertically Stacked Tensor:\n", vstacked_tensor)
print("Horizontally Stacked Tensor:\n", hstacked_tensor)

# Squeezing - removes all 1 dimensions from a tensor
# Unsqueezing - adds a 1 dimension to a tensor at a specified position
unsqueezed_tensor = torch.tensor([[[1,2,3]]])  # Shape (1,1,3)
squeezed_tensor = torch.squeeze(unsqueezed_tensor)
print("Unsqueezed Tensor Shape:", unsqueezed_tensor.shape)
print("Squeezed Tensor Shape:", squeezed_tensor.shape)

# Permuting - rearranges the dimensions of a tensor
permuted_tensor = torch.randn(2,3,4)  # Shape (2,3,4)
permuted_tensor = torch.permute(permuted_tensor, (1,0,2))  # New shape (3,2,4)
print("Permuted Tensor Shape:", permuted_tensor.shape)


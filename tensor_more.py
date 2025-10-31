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

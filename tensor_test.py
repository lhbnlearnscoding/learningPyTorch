import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.rand(2, 2)
print("x =", x)
print("y =", y)
print("x+y =", x+y)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
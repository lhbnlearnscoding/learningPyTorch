import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**3 - 4*x**2 + x
y.backward()
print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())
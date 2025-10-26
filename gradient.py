import torch
torch.manual_seed(0) # for reproducibility

x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x
y.backward() # compute dy/dx 

print("x grad:", x.grad)  # should print dy/dx at x=2.0




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ===== 0) Cấu hình chung =====
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1) Tạo dữ liệu giả: y = 3x - 2 + noise =====
N = 500
X = torch.randn(N, 1)
Y = 3 * X - 2 + 0.2 * torch.randn(N, 1)

# Chia train/test (80/20)
n_train = int(0.8 * N)
X_train, Y_train = X[:n_train], Y[:n_train]
X_test,  Y_test  = X[n_train:], Y[n_train:]

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, Y_test),   batch_size=64, shuffle=False)

# ===== 2) Model, Loss, Optimizer =====
model = nn.Linear(1, 1).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

# ===== 3) Train =====
EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        yhat = model(xb)
        loss = criterion(yhat, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    if epoch % 20 == 0 or epoch == 1:
        avg_loss = total_loss / len(train_loader.dataset)
        w = model.weight.item()
        b = model.bias.item()
        print(f"[Epoch {epoch:3d}] train MSE={avg_loss:.6f} | w≈{w:.3f}, b≈{b:.3f}")

# ===== 4) Đánh giá =====
model.eval()
with torch.no_grad():
    se_sum, n = 0.0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        yhat = model(xb)
        se_sum += ((yhat - yb) ** 2).sum().item()
        n += xb.size(0)
    mse = se_sum / n

w = model.weight.item()
b = model.bias.item()
print(f"\nKết quả cuối: w≈{w:.3f}, b≈{b:.3f} (kỳ vọng: w≈3, b≈-2)")
print(f"Test MSE: {mse:.6f}")

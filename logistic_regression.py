import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ===== 0) Cấu hình chung =====
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1) Tạo dữ liệu phân loại nhị phân 2D (2 cụm điểm) =====
N = 600
N_per_class = N // 2

# class 0 quanh (-1, -1); class 1 quanh (1, 1)
X0 = torch.randn(N_per_class, 2) * 0.9 + torch.tensor([-1.0, -1.0])
X1 = torch.randn(N_per_class, 2) * 0.9 + torch.tensor([ 1.0,  1.0])
Y0 = torch.zeros(N_per_class, 1)
Y1 = torch.ones(N_per_class, 1)

X = torch.cat([X0, X1], dim=0)
Y = torch.cat([Y0, Y1], dim=0)

# Trộn dữ liệu
perm = torch.randperm(N)
X, Y = X[perm], Y[perm]

# Chia train/test (80/20)
n_train = int(0.8 * N)
X_train, Y_train = X[:n_train], Y[:n_train]
X_test,  Y_test  = X[n_train:], Y[n_train:]

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  Y_test),  batch_size=64, shuffle=False)

# ===== 2) Model logistic: Linear(2->1) + BCEWithLogitsLoss =====
model = nn.Linear(2, 1).to(device)
criterion = nn.BCEWithLogitsLoss()  # ổn định số hơn so với Sigmoid + BCELoss
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

# ===== 3) Train =====
EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)             # (batch,1)
        loss = criterion(logits, yb)   # yb ∈ {0,1}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    if epoch % 20 == 0 or epoch == 1:
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch:3d}] train BCE={avg_loss:.6f}")

# ===== 4) Đánh giá: accuracy =====
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        correct += (pred == yb).sum().item()
        total   += yb.numel()

acc = correct / total
print(f"\nTest accuracy: {acc*100:.2f}%")

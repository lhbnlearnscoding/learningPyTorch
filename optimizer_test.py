import torch
import torch.nn as nn
import torch.optim as optim

# ===== 0) Fix seed cho tái lập kết quả =====
torch.manual_seed(0)

# ===== 1) Tạo dữ liệu giả: y = 3x - 2 + noise =====
N = 200
X = torch.randn(N, 1)
Y = 3 * X - 2 + 0.2 * torch.randn(N, 1)

# ===== 2) Hàm train chung =====
def train_once(optimizer_name="sgd", lr=0.1, epochs=200, use_scheduler=False):
    model = nn.Linear(1, 1)               # mô hình tuyến tính đơn giản
    criterion = nn.MSELoss()

    # Chọn optimizer
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)   # SGD + momentum
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)                 # Adam
    else:
        raise ValueError("optimizer_name phải là 'sgd' hoặc 'adam'")

    # (Tuỳ chọn) Scheduler giảm lr theo epoch
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    else:
        scheduler = None

    for epoch in range(1, epochs + 1):
        # 1) forward
        y_hat = model(X)
        loss = criterion(y_hat, Y)

        # 2) zero_grad -> 3) backward -> 4) step
        optimizer.zero_grad()
        loss.backward()

        # (tuỳ chọn) clip gradient để tránh nổ gradient với mô hình lớn
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # 5) scheduler (nếu có)
        if scheduler:
            scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            # Lấy lr hiện tại từ optimizer
            curr_lr = optimizer.param_groups[0]["lr"]
            w = model.weight.item()
            b = model.bias.item()
            print(f"[{optimizer_name.upper()}] epoch {epoch:3d} | lr={curr_lr:.4f} | loss={loss.item():.6f} | w={w:.3f}, b={b:.3f}")

    # In kết quả cuối
    w = model.weight.item()
    b = model.bias.item()
    print(f"--> {optimizer_name.upper()} kết thúc: w≈{w:.3f}, b≈{b:.3f}\n")
    return model

print("=== Train với SGD (momentum) ===")
train_once(optimizer_name="sgd", lr=0.1, epochs=200, use_scheduler=True)

print("=== Train với Adam ===")
train_once(optimizer_name="adam", lr=0.05, epochs=200, use_scheduler=False)

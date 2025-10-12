import os, shutil
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- cấu hình chung ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Nên đặt ROOT vào 1 thư mục "sạch" bạn có quyền ghi. Đổi nếu bạn muốn:
ROOT = r"./data"  # ví dụ: r"D:\datasets\cifar10"

def clean_cifar10_cache(root):
    """Delete old CIFAR-10 to redownload new torchvision."""
    shutil.rmtree(os.path.join(root, "cifar-10-batches-py"), ignore_errors=True)
    gz = os.path.join(root, "cifar-10-python.tar.gz")
    if os.path.exists(gz):
        os.remove(gz)

# --- transforms ---
tfm_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])
tfm_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
])

# --- model ---
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self,x): 
        return self.classifier(self.features(x))

def make_loaders():
    """Tạo train_loader và test_loader, có cơ chế retry nếu file tải bị hỏng."""
    try:
        train_set = datasets.CIFAR10(ROOT, train=True,  download=True, transform=tfm_train)
        test_set  = datasets.CIFAR10(ROOT, train=False, download=True, transform=tfm_test)
    except Exception as e:
        print("[WARN] Lỗi tải CIFAR-10:", e)
        print("[INFO] Xoá cache hỏng và thử tải lại...")
        clean_cifar10_cache(ROOT)
        train_set = datasets.CIFAR10(ROOT, train=True,  download=True, transform=tfm_train)
        test_set  = datasets.CIFAR10(ROOT, train=False, download=True, transform=tfm_test)

    # Lưu ý Windows: num_workers>0 cần đặt trong guard __main__
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader

def eval_loader(model, loader, criterion):
    model.eval(); correct=total=loss_sum=0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item()*xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred==yb).sum().item()
            total   += yb.size(0)
    return loss_sum/total, correct/total

def main():
    # 1) Data
    os.makedirs(ROOT, exist_ok=True)
    train_loader, test_loader = make_loaders()

    # 2) Model + loss + optim + scheduler
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 3) Train + Eval
    EPOCHS = 20
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        test_loss, test_acc = eval_loader(model, test_loader, criterion)
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:02d} | lr={curr_lr:.6f} | test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")

    torch.save(model.state_dict(), "cifar10_cnn.pt")
    print("Đã lưu:", os.path.abspath("cifar10_cnn.pt"))

if __name__ == "__main__":
    main()

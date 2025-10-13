import os, shutil
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ==== ĐỔI PATH Ở ĐÂY (giống bài trước) ====
ROOT = r"./data"   # ví dụ: r"D:\datasets\cifar10"

def clean_cifar10_cache(root):
    shutil.rmtree(os.path.join(root, "cifar-10-batches-py"), ignore_errors=True)
    gz = os.path.join(root, "cifar-10-python.tar.gz")
    if os.path.exists(gz): os.remove(gz)

# transforms: train (augment) / test (no augment)
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

def make_loaders():
    os.makedirs(ROOT, exist_ok=True)
    try:
        train_set = datasets.CIFAR10(ROOT, train=True,  download=True, transform=tfm_train)
        test_set  = datasets.CIFAR10(ROOT, train=False, download=True, transform=tfm_test)
    except Exception as e:
        print("[WARN] Lỗi tải CIFAR-10:", e)
        print("[INFO] Xoá cache và thử lại...")
        clean_cifar10_cache(ROOT)
        train_set = datasets.CIFAR10(ROOT, train=True,  download=True, transform=tfm_train)
        test_set  = datasets.CIFAR10(ROOT, train=False, download=True, transform=tfm_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader

def main():
    train_loader, test_loader = make_loaders()

    # ===== Model: ResNet18 pretrained + head 10 lớp =====
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net = net.to(device)

    # (Tuỳ chọn) ĐÓNG BĂNG backbone 5 epoch đầu
    for name, p in net.named_parameters():
        if not name.startswith("fc"):
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    head_params = [p for n,p in net.named_parameters() if n.startswith("fc") and p.requires_grad]
    backbone_params = [p for n,p in net.named_parameters() if (not n.startswith("fc")) and p.requires_grad]

    optimizer = optim.AdamW(
        [{"params": head_params, "lr": 3e-3, "weight_decay": 5e-4},
         {"params": backbone_params, "lr": 3e-4, "weight_decay": 5e-4}],
    )

    EPOCHS = 15
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[3e-3, 3e-4],
        epochs=EPOCHS, steps_per_epoch=steps_per_epoch
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def eval_loader():
        net.eval(); correct=total=loss_sum=0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = net(xb); loss = criterion(logits, yb)
                loss_sum += loss.item()*xb.size(0)
                pred = logits.argmax(1)
                correct += (pred==yb).sum().item()
                total += yb.size(0)
        return loss_sum/total, correct/total

    for ep in range(1, EPOCHS+1):
        # MỞ KHOÁ backbone từ epoch 6 để fine-tune toàn bộ
        if ep == 6:
            for p in net.parameters(): p.requires_grad = True

        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = net(xb); loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        test_loss, test_acc = eval_loader()
        lr_head = optimizer.param_groups[0]["lr"]
        lr_back = optimizer.param_groups[1]["lr"]
        print(f"Epoch {ep:02d} | lr_head={lr_head:.5f} lr_back={lr_back:.5f} | "
              f"test_loss={test_loss:.4f} | test_acc={test_acc*100:.2f}%")

    torch.save(net.state_dict(), "cifar10_resnet18_finetune.pt")
    print("Saved:", os.path.abspath("cifar10_resnet18_finetune.pt"))

if __name__ == "__main__":  # bắt buộc khi num_workers>0 trên Windows
    main()

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0) # for reproducibility

# 1) Data
tfm = transforms.Compose([
    transforms.ToTensor(),                         # [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),    # mean/std MNIST
])
train_full = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
n_val = 5000
train_set, val_set = random_split(train_full, [len(train_full)-n_val, n_val])
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=256)
test_loader  = DataLoader(test_set,  batch_size=256)

# 2) Model: 28x28=784 -> 256 -> 128 -> 10
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

model = MLP().to(device) # chuyển model lên GPU nếu có
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 

def eval_loader(loader):
    model.eval(); correct=total=loss_sum=0.0
    with torch.no_grad(): # không tính gradient với no_grad
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item()*xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred==yb).sum().item()
            total   += yb.size(0)
    return loss_sum/total, correct/total

# 3) Train
EPOCHS=10
for ep in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    scheduler.step()
    val_loss, val_acc = eval_loader(val_loader)
    print(f"Epoch {ep:02d} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

# 4) Test + save
test_loss, test_acc = eval_loader(test_loader)
print(f"TEST  | loss={test_loss:.4f} | acc={test_acc*100:.2f}%")
torch.save(model.state_dict(), "mnist_mlp.pt")

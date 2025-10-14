import os, io, zipfile, urllib.request, shutil, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

# ==================== Config ====================
DATA_ROOT = Path("./data_pennfudan")  # đổi nếu muốn
BATCH_SIZE = 2
NUM_WORKERS = 2
EPOCHS = 10
BASE_LR = 5e-4
WARMUP_ITERS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ==================== 0) Download & extract Penn-Fudan ====================
# Official URL from UPenn (≈51MB zip)
PENN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

def ensure_dataset(root: Path):
    imgs_dir = root / "PennFudanPed" / "PNGImages"
    ann_dir  = root / "PennFudanPed" / "PedMasks"
    if imgs_dir.exists() and ann_dir.exists():
        print("[OK] Penn-Fudan already present at", root)
        return
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "PennFudanPed.zip"
    print("[DL] Downloading Penn-Fudan to", zip_path)
    with urllib.request.urlopen(PENN_URL) as r:
        data = r.read()
    with open(zip_path, "wb") as f:
        f.write(data)
    print("[EXTRACT] Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)
    zip_path.unlink(missing_ok=True)
    print("[OK] Dataset ready:", root)

# ==================== 1) Dataset ====================
class PennFudanDet(Dataset):
    """
    Trả về (image, target) cho Faster R-CNN.
    - image: Tensor float [3,H,W] trong [0,1]
    - target: dict {
        boxes: FloatTensor [N,4] (xmin,ymin,xmax,ymax),
        labels: Int64Tensor [N] (1 = pedestrian),
        image_id: Int64Tensor [1],
        area: FloatTensor [N],
        iscrowd: Int64Tensor [N] (0)
      }
    """
    def __init__(self, root: Path, train=True):
        self.root = root / "PennFudanPed"
        self.imgs = sorted((self.root / "PNGImages").glob("*.png"))
        self.masks = sorted((self.root / "PedMasks").glob("*.png"))
        assert len(self.imgs) == len(self.masks) and len(self.imgs) > 0
        # split 120/50 (train/val) đơn giản
        split = int(0.7 * len(self.imgs))
        if train:
            self.imgs = self.imgs[:split]
            self.masks = self.masks[:split]
        else:
            self.imgs = self.imgs[split:]
            self.masks = self.masks[split:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])  # mỗi người = 1 id khác nhau (>0)

        img = TF.to_tensor(img)  # [0,1], CxHxW
        mask_np = np.array(mask, dtype=np.int32)

        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids != 0]  # 0 là nền
        boxes = []
        for oid in obj_ids:
            ys, xs = np.where(mask_np == oid)
            if ys.size == 0 or xs.size == 0:  # an toàn
                continue
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 1 = pedestrian
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        # Augment đơn giản (train): horizontal flip p=0.5
        if self.training and torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            w = img.shape[2]
            boxes_flipped = boxes.clone()
            boxes_flipped[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes_flipped

        return img, target

# collate_fn để DataLoader gom list (vì detection dùng list per-image)
def collate_fn(batch):
    return tuple(zip(*batch))

# ==================== 2) Model ====================
def create_model(num_classes=2):
    # num_classes = 2 (background + pedestrian)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

# ==================== 3) Utils ====================
@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, loss_sum = 0, 0.0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, tgts)  # in eval mode vẫn trả loss nếu có targets
        loss = sum(v.item() for v in loss_dict.values())
        loss_sum += loss * len(images)
        total += len(images)
    return loss_sum / max(total, 1)

def warmup_lr_lambda(it):
    # warmup tuyến tính cho WARMUP_ITERS đầu; sau đó LR = 1.0
    if it >= WARMUP_ITERS:
        return 1.0
    return float(it) / float(max(1, WARMUP_ITERS))

# ==================== 4) Train ====================
def main():
    ensure_dataset(DATA_ROOT)
    train_set = PennFudanDet(DATA_ROOT, train=True)
    val_set   = PennFudanDet(DATA_ROOT, train=False)
    # set cờ training cho augment trong __getitem__
    train_set.training = True
    val_set.training = False

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = create_model(num_classes=2).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=BASE_LR, momentum=0.9, weight_decay=5e-4)

    # warmup + cosine decay theo epochs
    total_iters = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

    print(f"[INFO] Train imgs: {len(train_set)} | Val imgs: {len(val_set)} | Device: {DEVICE.type}")
    global_iter = 0
    best_val = math.inf
    save_path = "fasterrcnn_pennfudan.pt"

    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for images, targets in train_loader:
            images = [im.to(DEVICE) for im in images]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, tgts)      # trả dict loss
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # LR schedule per-iter: warmup rồi cosine
            if global_iter < WARMUP_ITERS:
                warmup.step()
            else:
                scheduler.step()
            global_iter += 1

        val_loss = evaluate_loss(model, val_loader, DEVICE)
        elapsed = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:02d} | lr={curr_lr:.6f} | val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ↳ Saved best to {save_path}")

    print("[DONE] Best val_loss:", best_val)

if __name__ == "__main__":
    # Windows lưu ý: nếu tăng NUM_WORKERS>0, cần guard như thế này
    main()

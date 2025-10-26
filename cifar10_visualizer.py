# Script dùng để nhìn trực quan (visualize) các phép tăng cường dữ liệu (data augmentation) trên ảnh CIFAR-10. 
# Nó không huấn luyện mô hình; chỉ tải vài ảnh mẫu rồi áp các biến đổi và lưu ảnh trước/sau thành các tấm grid để bạn so sánh.
import os
from pathlib import Path
import torch
from torchvision import datasets, transforms, utils as vutils

# ===== Cấu hình =====
ROOT = "./data"
OUT = Path("./aug_samples")
N_SAMPLES = 8  # số ảnh gốc để minh hoạ trên mỗi grid

torch.manual_seed(0)
OUT.mkdir(parents=True, exist_ok=True)

# ===== Tải CIFAR-10 (chỉ cần ToTensor cho ảnh gốc) =====
base_tfm = transforms.ToTensor()
dataset = datasets.CIFAR10(ROOT, train=True, download=True, transform=base_tfm)
classes = dataset.classes
print("Classes:", classes)

# Lấy N_SAMPLES ảnh đầu tiên (đa lớp ngẫu nhiên)
images = []
labels = []
for i in range(N_SAMPLES):
    x, y = dataset[i]
    images.append(x)   # Tensor [3,32,32] trong [0,1]
    labels.append(classes[y])
grid_orig = vutils.make_grid(torch.stack(images, dim=0), nrow=N_SAMPLES, padding=2)
vutils.save_image(grid_orig, OUT / "00_original.png")
print("[Saved]", OUT / "00_original.png")

# ===== 1) RandomCrop(32, padding=4) =====
tfm_crop = transforms.Compose([
    transforms.Pad(4, padding_mode="reflect"),
    transforms.RandomCrop(32),
])
imgs_crop = [tfm_crop(img) for img in images]
grid_crop = vutils.make_grid(torch.stack(imgs_crop), nrow=N_SAMPLES, padding=2)
vutils.save_image(grid_crop, OUT / "01_randomcrop_pad4.png")
print("[Saved]", OUT / "01_randomcrop_pad4.png")

# ===== 2) RandomHorizontalFlip(p=0.5) =====
tfm_flip = transforms.RandomHorizontalFlip(p=1.0)  # ép flip để thấy rõ
imgs_flip = [tfm_flip(img) for img in images]
grid_flip = vutils.make_grid(torch.stack(imgs_flip), nrow=N_SAMPLES, padding=2)
vutils.save_image(grid_flip, OUT / "02_hflip.png")
print("[Saved]", OUT / "02_hflip.png")

# ===== 3) ColorJitter (độ sáng/độ tương phản/màu/sắc độ) =====
tfm_cj = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)
imgs_cj = [torch.clamp(tfm_cj(img), 0.0, 1.0) for img in images]
grid_cj = vutils.make_grid(torch.stack(imgs_cj), nrow=N_SAMPLES, padding=2)
vutils.save_image(grid_cj, OUT / "03_colorjitter.png")
print("[Saved]", OUT / "03_colorjitter.png")

# ===== 4) RandomGrayscale(p=0.5) =====
tfm_gray = transforms.RandomGrayscale(p=1.0)  # ép grayscale để dễ quan sát
imgs_gray = [tfm_gray(img) for img in images]
grid_gray = vutils.make_grid(torch.stack(imgs_gray), nrow=N_SAMPLES, padding=2)
vutils.save_image(grid_gray, OUT / "04_grayscale.png")
print("[Saved]", OUT / "04_grayscale.png")

# ===== 5) RandomErasing (Cutout) =====
# RandomErasing thường áp trên Tensor sau Normalize; ở đây demo trực tiếp để dễ nhìn
eraser = transforms.RandomErasing(p=1.0, scale=(0.06, 0.12), ratio=(0.5, 2.0), value=0.0)
imgs_erase = []
for img in images:
    im = img.clone()
    imgs_erase.append(eraser(im))
grid_erase = vutils.make_grid(torch.stack(imgs_erase), nrow=N_SAMPLES, padding=2)
vutils.save_image(grid_erase, OUT / "05_randomerasing.png")
print("[Saved]", OUT / "05_randomerasing.png")

print("\nAll samples saved to:", OUT.resolve())
print("Tip: mở 00_original.png và các file còn lại để so sánh từng augmentation.")

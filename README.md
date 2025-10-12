# learningPyTorch_from Basics to CNNs & Transfer Learning
🎯 Goals

Master the standard pipeline: Data → Transforms → DataLoader → Model → Loss/Optim → Train/Eval → Save

Hands-on projects: Linear/Logistic Regression, MNIST (MLP), CIFAR-10 (CNN), ResNet18 Pretrained (Transfer Learning).

📦 Environment & Setup
# (recommended) conda
conda create -n pytorch_env python=3.10 -y
conda activate pytorch_env

# CPU
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# or GPU (pick the right CUDA version for your machine)
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install matplotlib tqdm

🧰 Datasets

MNIST & CIFAR-10 will auto-download into ./data by default.

You can change the data path by editing the ROOT variable in the scripts (e.g., r"D:\datasets\cifar10").

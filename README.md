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

💡 Technical Notes

Normalize

MNIST (1 channel): Normalize((0.1307,), (0.3081,))

CIFAR-10 (RGB): Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))

Train/Eval modes: use model.train() during training; model.eval() + torch.no_grad() for evaluation.

Weight Decay (AdamW): regularization to reduce overfitting; typically exclude bias and (Batch/Layer)Norm from decay.

📜 License

MIT

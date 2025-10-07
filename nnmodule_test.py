import torch
import torch.nn as nn

# 1. Định nghĩa một mạng nơ-ron cơ bản
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Mạng: 10 input → 5 hidden → 1 output
        self.fc1 = nn.Linear(10, 5)  # tầng fully connected 1
        self.relu = nn.ReLU()        # hàm kích hoạt ReLU
        self.fc2 = nn.Linear(5, 1)   # tầng fully connected 2

    def forward(self, x):
        # Dữ liệu đi qua các tầng
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Tạo model
model = SimpleNN()
print("Cấu trúc mạng nơ-ron:")
print(model)

# 3. Tạo dữ liệu ngẫu nhiên để test
x = torch.randn(3, 10)   # batch size = 3, mỗi input có 10 features
y = model(x)             # output sẽ có 3 giá trị (batch_size=3, output=1)
print("\nInput shape:", x.shape)
print("Output shape:", y.shape)
print("Output values:\n", y)

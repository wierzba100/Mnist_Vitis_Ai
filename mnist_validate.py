import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# 1. Ustawienia
# ==============================
batch_size = 64
model_path = "mnist_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Uzywane urzadzenie:", device)

# ==============================
# 2. Dane - MNIST (test)
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==============================
# 3. Definicja modelu (taka sama jak przy trenowaniu!)
# ==============================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==============================
# 4. Wczytanie modelu
# ==============================
model = Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model wczytany.")

# ==============================
# 5. Testowanie
# ==============================
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"Accuracy na zbiorze testowym: {accuracy:.2f}%")

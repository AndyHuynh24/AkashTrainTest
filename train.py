import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import time
import json

print("=" * 50)
print("MNIST Training Job")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Config
EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001
OUTPUT_DIR = "/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# Train
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

test_acc = 100. * correct / total
elapsed = time.time() - start_time

print("=" * 50)
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Training time: {elapsed:.1f}s")
print("=" * 50)

# Save
model_path = os.path.join(OUTPUT_DIR, "mnist_model.pt")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

results = {
    "test_accuracy": test_acc,
    "training_time_seconds": round(elapsed, 1),
    "epochs": EPOCHS,
    "device": str(device),
    "pytorch_version": torch.__version__
}
results_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")

print("\nDone!")

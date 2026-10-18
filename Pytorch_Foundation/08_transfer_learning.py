import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

model = resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images , labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss / len(train_loader):.4f}")

model.eval()
correct , total = 0 ,0
with torch.no_grad():
    for images, labels in test_loader:
        images , labels = images.to(device), labels.to(device)
        outputs = model(images)
        _ , predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nAccuracy on CIFAR-10 Test Set: {100 * correct / total:.2f}%")

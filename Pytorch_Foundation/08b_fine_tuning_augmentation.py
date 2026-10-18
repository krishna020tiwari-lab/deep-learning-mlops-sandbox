import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

train_transforms = transforms.Compose((
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224, padding=4, padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
))

test_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
     )]
)

train_set = torchvision.datasets.CIFAR10(root='./data',train=True,transform=train_transforms,download=True)

test_set = torchvision.datasets.CIFAR10(root='./data',train=False,transform=test_transforms,download=True)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=True)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

for name , param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

epochs = 3
print(f"\nFine-tuning ResNet-18 with Augmentation on CIFAR-10...\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images , labels = images.to(device) , labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss / len(train_loader):.4f}")

model.eval()
correct , total =0,0
with torch.no_grad():
    for images , labels in train_loader:
        images , labels = images.to(device) , labels.to(device)
        outputs = model(images)
        _ , predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"\nAccuracy on CIFAR-10 Test Set (Fine-Tuned): {100 * correct / total:.2f}%")






# Import libraries
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from medmnist import PneumoniaMNIST
from torchvision.models import Inception_V3_Weights

#  Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#  Transforms (grayscale → 3-channel + aug + normalize)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#  Load PneumoniaMNIST
train_dataset = PneumoniaMNIST(split='train', transform=transform, download=True)
val_dataset = PneumoniaMNIST(split='val', transform=transform, download=True)
test_dataset = PneumoniaMNIST(split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#  Load pretrained InceptionV3 + modify final FC layer
weights = Inception_V3_Weights.DEFAULT
model = models.inception_v3(weights=weights, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

#  Handle class imbalance
labels = np.array([label[0] for label in train_dataset.labels])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

#  Loss & optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#  Training function
def train(model, loader):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):  # (main_logits, aux_logits)
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

#  Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

#  Train
for epoch in range(10):
    train_loss = train(model, train_loader)
    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

#  Save trained model
torch.save(model.state_dict(), "inception_pneumonia.pth")
print(" Model saved as 'inception_pneumonia.pth'")

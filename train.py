import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from medmnist import INFO, PneumoniaMNIST
import medmnist
import numpy as np
from tqdm import tqdm

# Early Stopping class
def early_stopping(val_loss, best_loss, counter, patience=5):
    if best_loss is None or val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        return True, best_loss, counter
    return False, best_loss, counter

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loading
data_flag = 'pneumoniamnist'
download = True
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Data Transform with stronger augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load datasets
train_dataset = DataClass(split='train', download=download, transform=transform)
val_dataset = DataClass(split='val', download=download, transform=transform)
test_dataset = DataClass(split='test', download=download, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pretrained ResNet50
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Modify input and final layer
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# Compute class weights to handle imbalance
class_counts = np.bincount(train_dataset.labels.flatten())
class_weights = 1. / class_counts
weights = torch.FloatTensor(class_weights).to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training function
def train(model, loader):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    avg_loss = running_loss / len(loader)
    return acc, prec, recall, f1, avg_loss

# Training loop with Early Stopping
num_epochs = 50
best_loss = None
patience = 5
counter = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    loss = train(model, train_loader)
    acc, prec, recall, f1, val_loss = evaluate(model, val_loader)
    print(f"Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
    scheduler.step(val_loss)

    stop, best_loss, counter = early_stopping(val_loss, best_loss, counter, patience=patience)
    if stop:
        print("Early stopping triggered!")
        break

# Final test evaluation
test_acc, test_prec, test_recall, test_f1, test_loss = evaluate(model, test_loader)
print(f"\n🔍 Test Performance:\nAccuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_recall:.4f} | F1-score: {test_f1:.4f}")

# Save model
torch.save(model.state_dict(), "resnet50_pneumonia.pth")

print("Model training complete and saved!")

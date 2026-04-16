import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights


# 1. DATA TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# 2. LOAD DATA
train_data = datasets.ImageFolder('dataset/train', transform=transform)
test_data = datasets.ImageFolder('dataset/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

print("Classes:", train_data.classes)

# 3. LOAD MODEL

model = models.mobilenet_v2(
    weights=MobileNet_V2_Weights.DEFAULT
)
# Modify final layer
model.classifier[1] = nn.Linear(model.last_channel, 1)

# 4. LOSS + OPTIMIZER
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. TRAINING LOOP
for epoch in range(15):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 6. SAVE MODEL
torch.save(model.state_dict(), "model/cataract_model.pth")

# 7. TEST MODEL
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5

        correct += (preds.squeeze() == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.2f}")
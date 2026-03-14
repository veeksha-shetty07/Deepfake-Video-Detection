import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ===== SETTINGS =====
DATA_DIR = "faces"
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.0001
MODEL_PATH = "deepfake_model.pth"

# ===== IMAGE TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== LOAD DATA =====
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# ===== LOAD PRETRAINED MODEL =====
model = models.resnet18(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# ===== LOSS & OPTIMIZER =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== TRAINING =====
print("🚀 Training started...")
for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Training finished & model saved!")
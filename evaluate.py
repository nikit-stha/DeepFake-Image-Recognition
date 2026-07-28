import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms

from model import DualAttentionEnhancedNetwork

device = "cuda"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root="dataset/Dataset/Test", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

efficientnet = models.efficientnet_b3(weights=None)
features = efficientnet.features

model = DualAttentionEnhancedNetwork(
    shallow_stage=nn.Sequential(*list(features)[:3]),
    deep_stage=nn.Sequential(*list(features)[3:]),
    shallow_channels=32,
    deep_channels=1536,
    num_classes=2
).to(device)

model.load_state_dict(torch.load("deepfake_recognition.pth", map_location=device, weights_only=True))
model.eval()

correct = total = 0
with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
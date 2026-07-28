import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms

from model import DualAttentionEnhancedNetwork

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = "cuda"
path_file = "dataset/Dataset/Train"

efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
features = efficientnet.features
    
shallow_stage = nn.Sequential(*list(features)[:3])
deep_stage = nn.Sequential(*list(features)[3:])
    
model = DualAttentionEnhancedNetwork(
        shallow_stage=shallow_stage,
        deep_stage=deep_stage,
        shallow_channels=32,
        deep_channels=1536,
        num_classes=10
    ).to(device=device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params= model.parameters(), lr=0.001)


train_dataset = datasets.ImageFolder(root=path_file, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=25,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for epoch in range(3):
        running_loss = 0.0
        print(f'Epoch : {epoch}')
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            prediction = model(inputs)

            loss = loss_function(prediction, labels)
            optimizer.zero_grad()

            loss = loss_function(prediction, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Loss : {running_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'deepfake_recognition.pth')
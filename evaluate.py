import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms

from model import QFENet

device = "cuda"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if __name__ == "__main__":

    dataset = datasets.ImageFolder(
        root="dataset/Dataset/Test",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=40,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Classes:", dataset.class_to_idx)
    print("Number of test images:", len(dataset))


    efficientnet = models.efficientnet_b3(weights=None)
    features = efficientnet.features

    model = QFENet(
        shallow_stage=nn.Sequential(*list(features)[:3]),
        deep_stage=nn.Sequential(*list(features)[3:]),
        shallow_channels=32,
        deep_channels=1536,
        num_classes=1
    ).to(device)


    model.load_state_dict(
        torch.load(
            "deepfake_recognition.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Convert logits -> probabilities
            probabilities = torch.sigmoid(outputs)

            # Binary prediction threshold
            predictions = (probabilities > 0.5).long().squeeze(1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
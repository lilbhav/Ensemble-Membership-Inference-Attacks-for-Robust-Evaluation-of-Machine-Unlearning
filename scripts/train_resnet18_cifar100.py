"""
Train and save a ResNet-18 model on CIFAR-100.
This script trains a pre-trained ResNet-18 on CIFAR-100 and saves the checkpoint.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import get_num_classes


def train_resnet18_cifar100(
    num_epochs=12,
    batch_size=128,
    learning_rate=0.1,
    output_checkpoint="/content/gdrive/My Drive/mia_checkpoints/pretrained_cifar100.pt"
):
    """
    Train ResNet-18 on CIFAR-100 and save checkpoint.
    """

    # ========== SETUP ==========
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== LOAD DATA ==========
    print("\n1. Loading CIFAR-100 dataset...")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    train_dataset = datasets.CIFAR100(
        root="./data/raw",
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR100(
        root="./data/raw",
        train=False,
        transform=test_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"   Train set: {len(train_dataset)} images")
    print(f"   Test set: {len(test_dataset)} images")

    # ========== LOAD MODEL ==========
    print("\n2. Loading pre-trained ResNet-18...")
    num_classes = get_num_classes("cifar100")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Adapt the model for CIFAR-100 (modify first conv layer and classification head)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small CIFAR images
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    print(f"   ResNet-18 adapted for CIFAR-100 ({num_classes} classes)")

    # ========== SETUP TRAINING ==========
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ========== TRAINING LOOP ==========
    print(f"\n3. Training for {num_epochs} epochs...")
    best_acc = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{train_loss / train_total:.3f}",
                "acc": f"{100 * train_correct / train_total:.1f}%",
            })

        scheduler.step()

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        train_acc = 100 * train_correct / train_total

        print(f"   Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), output_checkpoint)
            print(f"   Checkpoint saved (acc: {test_acc:.2f}%)")

    print("\n4. Training complete!")
    print(f"   Best test accuracy: {best_acc:.2f}%")
    print(f"   Checkpoint saved to: {output_checkpoint}")

    return model


if __name__ == "__main__":
    model = train_resnet18_cifar100(
        num_epochs=12,
        batch_size=256,
        learning_rate=0.1,
        output_checkpoint="/content/gdrive/My Drive/mia_checkpoints/pretrained_cifar100.pt",
    )

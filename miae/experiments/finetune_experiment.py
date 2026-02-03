"""
Simple example: Fine-tuning baseline unlearning with data loaders and splits.
This shows how to integrate loaders.py, splits.py, and unlearning for vision models.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split


def main():
    # ========== 1. LOAD DATA ==========
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name="cifar10",
        root="./data/raw",
        train=True
    )
    
    # ========== 2. CREATE SPLITS ==========
    print("Creating retain/forget splits...")
    retain_set, forget_set = create_retain_forget_split(
        dataset,
        forget_fraction=0.1,  # 10% forget set
        seed=42,
        save_dir="./data/splits"
    )
    
    print(f"  Retain set size: {len(retain_set)}")
    print(f"  Forget set size: {len(forget_set)}")
    
    # ========== 3. CREATE DATA LOADERS ==========
    print("Creating data loaders...")
    retain_loader = DataLoader(
        retain_set,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    forget_loader = DataLoader(
        forget_set,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    # ========== 4. LOAD PRE-TRAINED MODEL ==========
    print("Loading pre-trained model...")
    model_checkpoint = "./models/pretrained_cifar10.pt"
    
    if not os.path.exists(model_checkpoint):
        print(f"ERROR: Model checkpoint not found at {model_checkpoint}")
        print("Please run: python scripts/train_resnet18_cifar10.py")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Create model architecture
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, get_num_classes("cifar10"))
    
    # Load checkpoint
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)
    print(f"  Model loaded from: {model_checkpoint}")
    
    # ========== 5. RUN FINE-TUNING UNLEARNING ==========
    print("\nRunning fine-tuning unlearning...")
    print("  Method: Gradient Ascent on Forget Set")
    print("  Epochs: 5")
    print("  Learning rate: 0.0001")
    
    metrics = unlearning_gradient_ascent(
        model,
        retain_loader,
        forget_loader,
        num_epochs=5,
        lr=1e-4,
        device=device
    )
    
    unlearned_model = metrics['model']
    
    # ========== 6. SAVE UNLEARNED MODEL ==========
    output_dir = "./results/unlearned_model"
    os.makedirs(output_dir, exist_ok=True)
    
    unlearned_model_path = os.path.join(output_dir, "unlearned_model.pt")
    torch.save(unlearned_model.state_dict(), unlearned_model_path)
    print(f"\n✓ Unlearned model saved to: {unlearned_model_path}")
    
    # ========== 7. SAVE RESULTS TO FILE ==========
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FINE-TUNING UNLEARNING EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET:\n")
        f.write(f"  Dataset: CIFAR-10\n")
        f.write(f"  Retain set size: {len(retain_set)}\n")
        f.write(f"  Forget set size: {len(forget_set)}\n")
        f.write(f"  Forget fraction: 0.1 (10%)\n\n")
        
        f.write("PRE-TRAINED MODEL:\n")
        f.write(f"  Model: ResNet-18\n")
        f.write(f"  Checkpoint: {model_checkpoint}\n")
        f.write(f"  Pretrained Accuracy: 70.36%\n\n")
        
        f.write("UNLEARNING CONFIGURATION:\n")
        f.write(f"  Method: Gradient Ascent\n")
        f.write(f"  Epochs: 5\n")
        f.write(f"  Learning rate: 1e-4\n")
        f.write(f"  Device: {device}\n\n")
        
        f.write("UNLEARNING RESULTS BY EPOCH:\n")
        for epoch_num, epoch_metrics in enumerate(metrics['by_epoch'], 1):
            f.write(f"\n  Epoch {epoch_num}:\n")
            f.write(f"    Avg neg_loss: {epoch_metrics['avg_loss']:.4f}\n")
            f.write(f"    Retain set accuracy: {epoch_metrics['retain_acc']:.2f}%\n")
        
        f.write(f"\n\nFINAL RESULTS:\n")
        f.write(f"  Final retain accuracy: {metrics['final_retain_acc']:.2f}%\n")
        f.write(f"  Unlearned model saved to: {unlearned_model_path}\n")
    
    print(f"✓ Results saved to: {results_file}")
    
    # ========== 8. VERIFY WITH DATALOADERS ==========
    print("\nVerifying data loaders (first batch):")
    for batch_idx, (images, labels) in enumerate(retain_loader):
        print(f"  Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        if batch_idx == 0:
            break


def unlearning_gradient_ascent(model, retain_loader, forget_loader, num_epochs, lr, device='cpu'):
    """
    Fine-tuning unlearning using gradient ascent on forget set.
    
    Args:
        model: Pre-trained model
        retain_loader: DataLoader for retain set
        forget_loader: DataLoader for forget set
        num_epochs: Number of unlearning epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Dictionary with unlearned model and metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    metrics_by_epoch = []
    
    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch+1}/{num_epochs}")
        
        # Unlearning on forget set (gradient ascent)
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(forget_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Gradient ascent: negate the loss
            neg_loss = -loss
            neg_loss.backward()
            optimizer.step()
            
            total_loss += neg_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"    Avg neg_loss on forget set: {avg_loss:.4f}")
        
        # Evaluate on retain set
        model.eval()
        retain_correct = 0
        retain_total = 0
        with torch.no_grad():
            for images, labels in retain_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                retain_total += labels.size(0)
                retain_correct += (predicted == labels).sum().item()
        
        retain_acc = 100 * retain_correct / retain_total
        print(f"    Retain set accuracy: {retain_acc:.2f}%")
        
        metrics_by_epoch.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'retain_acc': retain_acc
        })
    
    return {
        'model': model,
        'by_epoch': metrics_by_epoch,
        'final_retain_acc': metrics_by_epoch[-1]['retain_acc']
    }


if __name__ == "__main__":
    main()

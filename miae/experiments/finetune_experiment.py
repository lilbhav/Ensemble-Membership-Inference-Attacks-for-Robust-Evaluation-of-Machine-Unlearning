"""
Simple example: Fine-tuning baseline unlearning with data loaders and splits.
This shows how to integrate loaders.py, splits.py, and fine_tune.py together.
"""

import torch
from torch.utils.data import DataLoader
from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split
from miae.unlearning.fine_tune import FineTuneUnlearning


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
        num_workers=4
    )
    forget_loader = DataLoader(
        forget_set,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # ========== 4. CONFIGURE UNLEARNING ==========
    config = {
        "script_path": "Third_Party_Code/FT/baselines.py",
        "num_epochs": 5,
        "lr": 1e-4,
        "batch_size": 32,
    }
    
    # ========== 5. RUN FINE-TUNING UNLEARNING ==========
    print("Running fine-tuning unlearning...")
    unlearner = FineTuneUnlearning(config)
    
    # Assuming you have a pre-trained model checkpoint
    model_checkpoint = "./models/pretrained_cifar10.pt"
    output_dir = "./results/unlearned_model"
    
    result = unlearner.run(
        model_checkpoint_path=model_checkpoint,
        dataset_name="cifar10",
        output_dir=output_dir
    )
    
    print(f"Unlearning complete! Model saved to: {result}")
    
    # ========== 6. (OPTIONAL) VERIFY WITH DATALOADERS ==========
    print("\nIterating through retain set (first batch):")
    for batch_idx, (images, labels) in enumerate(retain_loader):
        print(f"  Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        if batch_idx == 0:  # Just show first batch
            break


if __name__ == "__main__":
    main()

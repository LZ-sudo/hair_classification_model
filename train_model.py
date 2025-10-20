"""Main script to train the hair classifier"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.model import HairClassifier
from src.dataset import create_dataloaders
from src.train import train_epoch, validate, save_checkpoint, get_class_weights

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config['data_dir'], config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = HairClassifier(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss function with optional class weighting
    if config.get('use_class_weights', False):
        print("\nCalculating class weights...")
        class_weights = get_class_weights(train_loader, config['num_classes'], device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
        print("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("Using standard CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])
    
    # Print augmentation settings
    print("\nAugmentation settings:")
    print(f"  Mixup: {'✓' if config.get('use_mixup', False) else '✗'}")
    print(f"  CutMix: {'✓' if config.get('use_cutmix', False) else '✗'}")
    
    # Training loop
    best_val_acc = 0.0
    freeze_epochs = config.get('freeze_epochs', 5)
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("="*70)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-"*70)
        
        # Freeze/unfreeze backbone
        if epoch == 1 and freeze_epochs > 0:
            model.freeze_backbone()
            print("🔒 Backbone frozen - training head only")
        elif epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            print("🔓 Backbone unfrozen - training full model")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{config['checkpoint_dir']}/best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_acc, save_path)
            print(f"⭐ New best model! Val Acc: {val_acc:.2f}%")
    
    print("\n" + "="*70)
    print(f"✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("="*70)

if __name__ == '__main__':
    main()
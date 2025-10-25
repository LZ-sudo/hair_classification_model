import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from src.dataset import get_class_names

def get_class_weights(train_loader, num_classes, device='cpu', class_names=None):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        train_loader: Training dataloader
        num_classes: Number of classes
        device: Device to put weights on
        class_names: List of class names (optional, for display)
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = torch.zeros(num_classes)
    
    # Count samples per class
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Inverse frequency weighting
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    
    # Print statistics
    print(f"\nClass distribution in training set:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        class_name = class_names[i] if class_names else f"class_{i}"
        print(f"  {class_name}: {int(count)} samples (weight: {weight:.3f})")
    
    return class_weights.to(device)

def mixup_data(x, y, alpha=0.2):
    """
    Apply mixup augmentation
    
    Args:
        x: Input images
        y: Labels
        alpha: Mixup parameter
        
    Returns:
        Mixed inputs, label_a, label_b, lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation
    
    Args:
        x: Input images
        y: Labels
        alpha: CutMix parameter
        
    Returns:
        Mixed inputs, label_a, label_b, lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        config: Config dict with augmentation settings
        
    Returns:
        Average loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply MixUp or CutMix
        use_mixup = config.get('use_mixup', False) and np.random.rand() > 0.5
        use_cutmix = config.get('use_cutmix', False) and np.random.rand() > 0.5
        
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, alpha=config.get('mixup_alpha', 0.2)
            )
            
            # Forward pass
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
        elif use_cutmix:
            images, labels_a, labels_b, lam = cutmix_data(
                images, labels, alpha=config.get('cutmix_alpha', 1.0)
            )
            
            # Forward pass
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
        else:
            # Standard training
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, filename='best_model.pth'):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_acc: Validation accuracy
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, checkpoint_path)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")






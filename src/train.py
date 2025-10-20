"""Training utilities"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np

def get_class_weights(train_loader, num_classes=3, device='cpu'):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        train_loader: Training dataloader
        num_classes: Number of classes
        device: Device to put weights on
        
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
    classes = ['straight', 'wavy', 'curly']
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"  {classes[i]}: {int(count)} samples (weight: {weight:.3f})")
    
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

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Loss function for mixup/cutmix
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """
    Train for one epoch with optional mixup/cutmix
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        config: Configuration dict
        
    Returns:
        Average loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    use_mixup = config.get('use_mixup', False)
    use_cutmix = config.get('use_cutmix', False)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    cutmix_alpha = config.get('cutmix_alpha', 1.0)
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup or cutmix randomly
        if use_mixup and use_cutmix:
            # 50% chance for each
            if np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            else:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
        elif use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
        elif use_cutmix:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
        else:
            # Standard training
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/(pbar.n+1), 'acc': 100.*correct/total})
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """
    Validate model
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        
    Returns:
        Average loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_acc: Validation accuracy
        save_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, save_path)
    print(f"✓ Checkpoint saved to {save_path}")
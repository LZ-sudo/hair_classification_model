import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class HairDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Dataset that automatically detects classes from subdirectories
        
        Args:
            data_dir: Root directory (e.g., './data_hair_type' or './data_hair_color')
            split: 'train', 'val', or 'test'
            transform: Albumentations transform
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        
        # Automatically detect classes from subdirectories
        self.classes = self._detect_classes()
        
        # Load all image paths and labels
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def _detect_classes(self):
        """Automatically detect classes from subdirectories"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        # Get all subdirectories (these are the classes)
        classes = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])
        
        if len(classes) == 0:
            raise ValueError(f"No class subdirectories found in {self.data_dir}")
        
        return classes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label

def get_num_classes(data_dir):
    """
    Get number of classes by detecting subdirectories in train split
    
    Args:
        data_dir: Root data directory
        
    Returns:
        int: Number of classes
    """
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    
    return len(classes)

def get_class_names(data_dir):
    """
    Get class names by detecting subdirectories in train split
    
    Args:
        data_dir: Root data directory
        
    Returns:
        list: Sorted list of class names
    """
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    
    return classes

def get_transforms(img_size=224, is_training=True, config=None):
    """Get augmentation pipeline"""
    
    if is_training and config:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=config.get('aug_hflip', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=config.get('aug_rotate', 15),
                p=0.5
            ),
            A.RandomBrightnessContrast(p=config.get('aug_brightness', 0.5)),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_balanced_sampler(dataset):
    """
    Create sampler that balances class representation
    
    Args:
        dataset: HairDataset instance
        
    Returns:
        WeightedRandomSampler
    """
    # Count samples per class
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nCreating balanced sampler:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    # Calculate weights for each sample (inverse frequency)
    sample_weights = []
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        weight = 1.0 / class_counts[class_name]
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def create_dataloaders(data_dir, config):
    """Create train, val, test dataloaders"""
    
    # Get transforms
    train_transform = get_transforms(config['img_size'], is_training=True, config=config)
    val_transform = get_transforms(config['img_size'], is_training=False)
    
    # Create datasets
    train_dataset = HairDataset(data_dir, 'train', train_transform)
    val_dataset = HairDataset(data_dir, 'val', val_transform)
    test_dataset = HairDataset(data_dir, 'test', val_transform)
    
    # Print detected classes
    print(f"\nDetected {len(train_dataset.classes)} classes: {train_dataset.classes}")
    
    # Create train dataloader with optional balanced sampling
    if config.get('use_balanced_sampling', False):
        print("\nUsing balanced sampling for training...")
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,  # Use sampler instead of shuffle
            num_workers=config['num_workers'],
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ==================== DATA SPLITTING FUNCTIONS ====================

def collect_image_paths(input_dir, classes=None):
    """
    Collect all image paths from input directory
    Classes are auto-detected if not provided
    
    Args:
        input_dir: Directory containing class subdirectories
        classes: List of class names (auto-detected if None)
        
    Returns:
        dict: {class_name: [list of image paths]}
    """
    # Auto-detect classes if not provided
    if classes is None:
        classes = sorted([
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ])
        print(f"Auto-detected classes: {classes}")
    
    image_data = {cls: [] for cls in classes}
    
    for class_name in classes:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist. Skipping.")
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                image_data[class_name].append(img_path)
    
    return image_data

def split_data(image_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split image paths into train/val/test sets
    
    Args:
        image_data: dict {class_name: [image_paths]}
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: {'train': paths, 'val': paths, 'test': paths}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_name, image_paths in image_data.items():
        n_samples = len(image_paths)
        
        if n_samples == 0:
            print(f"Warning: No images found for class '{class_name}'")
            continue
        
        # First split: train vs (val + test)
        train_paths, temp_paths = train_test_split(
            image_paths,
            train_size=train_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(
            temp_paths,
            train_size=val_size,
            random_state=random_seed,
            shuffle=True
        )
        
        splits['train'].extend([(p, class_name) for p in train_paths])
        splits['val'].extend([(p, class_name) for p in val_paths])
        splits['test'].extend([(p, class_name) for p in test_paths])
    
    return splits

def copy_files_to_split_dirs(splits, output_dir, classes):
    """
    Copy files to train/val/test directory structure
    
    Args:
        splits: dict from split_data()
        output_dir: Output directory (e.g., './data_hair_type')
        classes: List of class names
    """
    # Create directory structure
    for split_name in ['train', 'val', 'test']:
        for class_name in classes:
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
    
    # Copy files
    for split_name, file_list in splits.items():
        print(f"\nCopying {split_name} set ({len(file_list)} images)...")
        
        for src_path, class_name in tqdm(file_list):
            # Get filename
            filename = os.path.basename(src_path)
            
            # Destination path
            dst_path = os.path.join(output_dir, split_name, class_name, filename)
            
            # Copy file
            shutil.copy2(src_path, dst_path)

def prepare_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, 
                   test_ratio=0.15, random_seed=42):
    """
    Complete pipeline to prepare dataset from input directory
    Classes are automatically detected from subdirectories
    
    Args:
        input_dir: Source directory with class subdirectories
        output_dir: Destination directory for train/val/test splits
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        random_seed: Random seed
    """
    print("="*60)
    print("PREPARING CLASSIFICATION DATASET")
    print("="*60)
    
    # Step 1: Auto-detect classes and collect all image paths
    print(f"\nStep 1: Auto-detecting classes and collecting images from {input_dir}")
    image_data = collect_image_paths(input_dir, classes=None)  # Auto-detect
    classes = list(image_data.keys())
    
    # Print statistics
    print("\nDataset statistics:")
    total_images = 0
    for class_name, paths in image_data.items():
        print(f"  {class_name}: {len(paths)} images")
        total_images += len(paths)
    print(f"  Total: {total_images} images across {len(classes)} classes")
    
    if total_images == 0:
        print("\nError: No images found! Please check your input directory.")
        return
    
    # Step 2: Split data
    print(f"\nStep 2: Splitting data (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})")
    splits = split_data(image_data, train_ratio, val_ratio, test_ratio, random_seed)
    
    print("\nSplit statistics:")
    for split_name, file_list in splits.items():
        print(f"  {split_name}: {len(file_list)} images")
        # Count per class
        class_counts = {}
        for _, class_name in file_list:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in class_counts.items():
            print(f"    - {class_name}: {count}")
    
    # Step 3: Copy files
    print(f"\nStep 3: Copying files to {output_dir}")
    copy_files_to_split_dirs(splits, output_dir, classes)
    
    print("\n" + "="*60)
    print("✓ Dataset preparation complete!")
    print("="*60)
    print(f"\nYou can now run: python train_model.py")
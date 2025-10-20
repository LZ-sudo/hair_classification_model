"""Inference utilities"""

import torch
from PIL import Image
import numpy as np
from src.dataset import get_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HairPredictor:
    """Simple predictor for hair classification"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.classes = ['straight', 'wavy', 'curly']
        self.transform = get_transforms(img_size=224, is_training=False)
    
    @torch.no_grad()
    def predict(self, image_path):
        """
        Predict hair type for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with 'class', 'confidence', and 'probabilities'
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        
        # Predict
        outputs = self.model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = probabilities.max(0)
        
        return {
            'class': self.classes[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                self.classes[i]: probabilities[i].item() 
                for i in range(len(self.classes))
            }
        }
    
    @torch.no_grad()
    def predict_with_tta(self, image_path, n_augmentations=5):
        """
        Predict with Test-Time Augmentation
        Averages predictions across multiple augmented versions
        
        Args:
            image_path: Path to image
            n_augmentations: Number of augmented predictions to average
            
        Returns:
            dict with averaged predictions
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Define TTA transforms
        tta_transforms = [
            # Original
            A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Horizontal flip
            A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Slight rotation left
            A.Compose([
                A.Resize(224, 224),
                A.Rotate(limit=(-10, -10), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Slight rotation right
            A.Compose([
                A.Resize(224, 224),
                A.Rotate(limit=(10, 10), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Center crop
            A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
        ]
        
        # Use only requested number of augmentations
        tta_transforms = tta_transforms[:n_augmentations]
        
        # Collect predictions from all augmentations
        all_probs = []
        for transform in tta_transforms:
            aug_image = transform(image=image)['image']
            aug_image = aug_image.unsqueeze(0).to(self.device)
            
            outputs = self.model(aug_image)
            probs = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs)
        
        # Average predictions across all augmentations
        avg_probs = torch.stack(all_probs).mean(dim=0)
        confidence, predicted = avg_probs.max(0)
        
        return {
            'class': self.classes[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                self.classes[i]: avg_probs[i].item() 
                for i in range(len(self.classes))
            }
        }
    
    @torch.no_grad()
    def predict_batch(self, image_paths, use_tta=False):
        """
        Predict for multiple images
        
        Args:
            image_paths: List of image paths
            use_tta: Whether to use test-time augmentation
            
        Returns:
            List of prediction dicts
        """
        results = []
        for img_path in image_paths:
            if use_tta:
                result = self.predict_with_tta(img_path)
            else:
                result = self.predict(img_path)
            results.append(result)
        return results
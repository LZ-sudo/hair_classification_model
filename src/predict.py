import torch
from PIL import Image
import numpy as np
from src.dataset import get_transforms, get_class_names
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HairPredictor:
    """
    Task-agnostic predictor that automatically detects classes
    Can be used for hair type, hair color, or any classification task
    """
    
    def __init__(self, model, device='cpu', classes=None):
        """
        Args:
            model: Trained model
            device: Device to run on
            classes: List of class names (optional, will be inferred if not provided)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.classes = classes  # Will be set when predict is called if None
        self.transform = get_transforms(img_size=224, is_training=False)
    
    def _set_classes_if_needed(self, data_dir=None):
        """Set classes if not already set"""
        if self.classes is None and data_dir is not None:
            self.classes = get_class_names(data_dir)
    
    @torch.no_grad()
    def predict(self, image_path, data_dir=None):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            data_dir: Optional data directory to infer classes from
            
        Returns:
            dict with 'class', 'confidence', and 'probabilities'
        """
        # Set classes if needed
        self._set_classes_if_needed(data_dir)
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        
        # Predict
        outputs = self.model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = probabilities.max(0)
        
        # If classes not set, use indices
        if self.classes is None:
            predicted_class = f"class_{predicted.item()}"
            prob_dict = {f"class_{i}": probabilities[i].item() 
                        for i in range(len(probabilities))}
        else:
            predicted_class = self.classes[predicted.item()]
            prob_dict = {self.classes[i]: probabilities[i].item() 
                        for i in range(len(self.classes))}
        
        return {
            'class': predicted_class,
            'confidence': confidence.item(),
            'probabilities': prob_dict
        }
    
    @torch.no_grad()
    def predict_with_tta(self, image_path, n_augmentations=5, data_dir=None):
        """
        Predict with Test-Time Augmentation
        
        Args:
            image_path: Path to image file
            n_augmentations: Number of augmented versions to average
            data_dir: Optional data directory to infer classes from
            
        Returns:
            dict with 'class', 'confidence', and 'probabilities'
        """
        # Set classes if needed
        self._set_classes_if_needed(data_dir)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # TTA transforms
        tta_transforms = [
            A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            for _ in range(n_augmentations)
        ]
        
        # Also include the original (non-augmented) transform
        tta_transforms.append(self.transform)
        
        # Collect predictions
        all_probs = []
        for transform in tta_transforms:
            aug_image = transform(image=image)['image']
            aug_image = aug_image.unsqueeze(0).to(self.device)
            
            outputs = self.model(aug_image)
            probabilities = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probabilities)
        
        # Average predictions
        avg_probs = torch.stack(all_probs).mean(0)
        confidence, predicted = avg_probs.max(0)
        
        # If classes not set, use indices
        if self.classes is None:
            predicted_class = f"class_{predicted.item()}"
            prob_dict = {f"class_{i}": avg_probs[i].item() 
                        for i in range(len(avg_probs))}
        else:
            predicted_class = self.classes[predicted.item()]
            prob_dict = {self.classes[i]: avg_probs[i].item() 
                        for i in range(len(self.classes))}
        
        return {
            'class': predicted_class,
            'confidence': confidence.item(),
            'probabilities': prob_dict
        }
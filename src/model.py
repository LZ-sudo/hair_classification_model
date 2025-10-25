"""DeiT and ConvNeXt models for hair classification"""

import torch
import torch.nn as nn
import timm

class HairClassifier(nn.Module):
    def __init__(self, model_name='convnext_tiny_in22k', num_classes=3, dropout=0.3, pretrained=True):
        super().__init__()
        
        # Load pretrained model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.model_name = model_name
        
        # Add dropout to classifier based on model architecture
        if 'convnext' in model_name.lower():
            # ConvNeXt has 'head' with fc layer
            if hasattr(self.model, 'head') and hasattr(self.model.head, 'fc'):
                in_features = self.model.head.fc.in_features
                self.model.head.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, num_classes)
                )
        elif 'deit' in model_name.lower() or 'vit' in model_name.lower():
            # ViT/DeiT models have 'head' as Linear layer
            if hasattr(self.model, 'head'):
                in_features = self.model.head.in_features
                self.model.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, num_classes)
                )
    
    def forward(self, x):
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all layers except classifier head"""
        for name, param in self.model.named_parameters():
            # Don't freeze the head (different names for different architectures)
            if not any(head_name in name for head_name in ['head', 'fc', 'classifier']):
                param.requires_grad = False
        
        print(f"Backbone frozen. Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
        
        print(f"Backbone unfrozen. Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

def load_model(checkpoint_path, num_classes=3, model_name=None):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        model_name: Model architecture name (optional, for flexibility)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to infer model_name from checkpoint if not provided
    if model_name is None:
        # You might need to save model_name in checkpoint for full flexibility
        # For now, we'll use a default
        model_name = 'convnext_tiny_in22k'
    
    model = HairClassifier(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
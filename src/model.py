"""DeiT model for hair classification"""

import torch
import torch.nn as nn
import timm

class HairClassifier(nn.Module):
    def __init__(self, model_name='deit_small_patch16_224', num_classes=3, dropout=0.3, pretrained=True):
        super().__init__()
        
        # Load pretrained DeiT
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Add dropout to classifier
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
            if 'head' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True

def load_model(checkpoint_path, num_classes=3):
    """Load trained model from checkpoint"""
    model = HairClassifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
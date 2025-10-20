"""DeiT model for hair classification"""

import torch
import torch.nn as nn
import timm

class HairClassifier(nn.Module):
    def __init__(self, model_name='deit_base_patch16_224', num_classes=3, dropout=0.3, pretrained=True):
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

def load_model(checkpoint_path, num_classes=3, model_name='deit_base_patch16_224', strict=True):
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        model_name: Model architecture name (must match the checkpoint!)
        strict: If True, requires exact match of all weights. If False, allows partial loading.
    """
    model = HairClassifier(model_name=model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load state dict - strict=True ensures full model loading
    if strict:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Checkpoint loaded successfully (all weights matched)")
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"\n⚠ Partial checkpoint loading:")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")
        print("  Note: This is expected when loading from a different architecture")

    return model
"""Evaluate trained model on test set"""

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from src.model import load_model
from src.dataset import create_dataloaders, HairDataset, get_num_classes, get_class_names
from src.predict import HairPredictor
import os

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect classes
    num_classes = get_num_classes(config['data_dir'])
    class_names = get_class_names(config['data_dir'])
    
    print(f"\n{'='*70}")
    print(f"TESTING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Data directory: {config['data_dir']}")
    print(f"Checkpoint: {config['checkpoint_dir']}/{config['test_model']}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"{'='*70}\n")
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(config['checkpoint_dir'], config['test_model'])
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(
        checkpoint_path,
        num_classes=num_classes,
        model_name=config['model_name']
    )
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = create_dataloaders(config['data_dir'], config)
    
    # Check if TTA is enabled
    use_tta = config.get('use_tta', False)
    
    if use_tta:
        print(f"\nEvaluating with Test-Time Augmentation ({config.get('tta_augmentations', 5)} augmentations)...")
        predictor = HairPredictor(model, device, classes=class_names)
        
        # Get test dataset to access image paths
        test_dataset = HairDataset(config['data_dir'], 'test', transform=None)
        
        all_preds = []
        all_labels = []
        
        # Predict with TTA for each image
        for img_path, label in tqdm(test_dataset.samples, desc='Testing'):
            result = predictor.predict_with_tta(
                img_path, 
                n_augmentations=config.get('tta_augmentations', 5),
                data_dir=config['data_dir']
            )
            pred_class = result['class']
            pred_idx = class_names.index(pred_class)
            
            all_preds.append(pred_idx)
            all_labels.append(label)
    
    else:
        print("\nEvaluating without TTA...")
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
    
    # Calculate metrics
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Overall accuracy
    accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # Confusion matrix
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print header
    header = "True \\ Pred"
    print(f"\n{header:<15}", end="")
    for class_name in class_names:
        print(f"{class_name:<15}", end="")
    print()
    print("-" * (15 + 15 * len(class_names)))
    
    # Print rows
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:<15}", end="")
        print()
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()






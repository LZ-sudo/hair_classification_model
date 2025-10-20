"""Evaluate trained model on test set"""

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from src.model import load_model
from src.dataset import create_dataloaders, HairDataset
from src.predict import HairPredictor

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = load_model(f"{config['checkpoint_dir']}/best_model.pth", config['num_classes'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = create_dataloaders(config['data_dir'], config)
    
    # Check if TTA is enabled
    use_tta = config.get('use_tta', False)
    
    if use_tta:
        print(f"\nEvaluating with Test-Time Augmentation ({config.get('tta_augmentations', 5)} augmentations)...")
        predictor = HairPredictor(model, device)
        
        # Get test dataset to access image paths
        test_dataset = HairDataset(config['data_dir'], 'test', transform=None)
        
        all_preds = []
        all_labels = []
        
        # Predict with TTA for each image
        for img_path, label in tqdm(test_dataset.samples, desc='Testing'):
            result = predictor.predict_with_tta(img_path, n_augmentations=config.get('tta_augmentations', 5))
            pred_class = result['class']
            pred_idx = ['straight', 'wavy', 'curly'].index(pred_class)
            
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
    
    # Calculate and display results
    classes = ['straight', 'wavy', 'curly']
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))
    
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("           ", "  ".join(f"{c:>8}" for c in classes))
    cm = confusion_matrix(all_labels, all_preds)
    for i, row in enumerate(cm):
        print(f"Actual {classes[i]:>8}", "  ".join(f"{val:>8}" for val in row))
    
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\n{'='*70}")
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    if use_tta:
        print(f"(with {config.get('tta_augmentations', 5)}-augmentation TTA)")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
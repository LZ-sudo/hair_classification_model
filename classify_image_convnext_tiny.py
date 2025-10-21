"""Classify hair type using ConvNeXt-Tiny model

Simple inference script specifically for ConvNeXt-Tiny trained model.

Usage:
    python classify_image_convnext_tiny.py --image path/to/image.jpg
    python classify_image_convnext_tiny.py --folder path/to/images/
"""

import argparse
import yaml
import torch
from src.model import HairClassifier
from src.predict import HairPredictor
import os
from glob import glob

def main():
    parser = argparse.ArgumentParser(description='Classify hair type using ConvNeXt-Tiny')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_convnext/best_model.pth',
                       help='Path to ConvNeXt-Tiny checkpoint')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please ensure you have trained the ConvNeXt-Tiny model first.")
        return
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ConvNeXt-Tiny model...")
    print(f"Device: {device}")
    
    # Load model
    model = HairClassifier(
        model_name='convnext_tiny',
        num_classes=3,
        dropout=0.2,
        pretrained=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    predictor = HairPredictor(model, device)
    
    # Predict
    if args.image:
        # Single image prediction
        print(f"\nClassifying: {args.image}")
        print("-" * 60)
        
        result = predictor.predict(args.image)
        
        print(f"Predicted class: {result['class'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll probabilities:")
        for cls, prob in result['probabilities'].items():
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {cls:>8}: {bar} {prob:.2%}")
    
    elif args.folder:
        # Folder of images
        image_paths = (glob(os.path.join(args.folder, '*.jpg')) + 
                      glob(os.path.join(args.folder, '*.png')) + 
                      glob(os.path.join(args.folder, '*.jpeg')))
        
        if not image_paths:
            print(f"No images found in {args.folder}")
            return
        
        print(f"\nFound {len(image_paths)} images")
        print("-" * 60)
        
        results = predictor.predict_batch(image_paths)
        
        # Display results
        for img_path, result in zip(image_paths, results):
            filename = os.path.basename(img_path)
            print(f"{filename:>30} → {result['class']:>8} ({result['confidence']:.1%})")
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        class_counts = {'straight': 0, 'wavy': 0, 'curly': 0}
        avg_confidence = 0
        
        for result in results:
            class_counts[result['class']] += 1
            avg_confidence += result['confidence']
        
        avg_confidence /= len(results)
        
        print(f"Total images: {len(results)}")
        print(f"Average confidence: {avg_confidence:.2%}")
        print(f"\nDistribution:")
        for cls, count in class_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {cls:>8}: {count:>3} ({percentage:.1f}%)")
    
    else:
        print("Error: Please provide either --image or --folder argument")
        print("\nExamples:")
        print("  python classify_image_convnext_tiny.py --image test.jpg")
        print("  python classify_image_convnext_tiny.py --folder test_images/")

if __name__ == '__main__':
    main()
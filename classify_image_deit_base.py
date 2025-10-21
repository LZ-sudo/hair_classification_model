"""Classify hair type in a single image or folder of images"""

import argparse
import yaml
import torch
from src.model import load_model
from src.predict import HairPredictor
import os
from glob import glob

def main():
    parser = argparse.ArgumentParser(description='Classify hair type')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, config['num_classes'])
    predictor = HairPredictor(model, device)
    
    # Predict
    if args.image:
        # Single image
        result = predictor.predict(args.image)
        print(f"\nImage: {args.image}")
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities: {result['probabilities']}")
    
    elif args.folder:
        # Folder of images
        image_paths = glob(os.path.join(args.folder, '*.jpg')) + \
                      glob(os.path.join(args.folder, '*.png')) + \
                      glob(os.path.join(args.folder, '*.jpeg'))
        
        print(f"Found {len(image_paths)} images")
        results = predictor.predict_batch(image_paths)
        
        for img_path, result in zip(image_paths, results):
            print(f"\n{os.path.basename(img_path)}: {result['class']} ({result['confidence']:.2%})")
    
    else:
        print("Please provide --image or --folder argument")

if __name__ == '__main__':
    main()
"""
Hair Color Classification with ConvNeXt-Tiny
Usage:
    python classify_image_convnext_hair_color.py --image path/to/image.jpg --checkpoint checkpoints_hair_color/best_model.pth
    python classify_image_convnext_hair_color.py --folder path/to/folder/ --checkpoint checkpoints_hair_color/best_model.pth
"""

import argparse
import torch
import os
from src.model import load_model
from src.predict import HairPredictor
from src.dataset import get_class_names

def classify_single_image(image_path, checkpoint_path, data_dir):
    """Classify a single hair image for color"""
    # Get class names
    class_names = get_class_names(data_dir)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        checkpoint_path,
        num_classes=len(class_names),
        model_name='convnext_tiny_in22k'  # or 'convnext_tiny'
    )
    
    # Create predictor
    predictor = HairPredictor(model, device, classes=class_names)
    
    # Predict
    result = predictor.predict(image_path, data_dir=data_dir)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"IMAGE: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    print(f"Predicted Color: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll Probabilities:")
    for class_name, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 40)
        print(f"  {class_name:15} {prob:6.2%} {bar}")
    print(f"{'='*60}\n")
    
    return result

def classify_folder(folder_path, checkpoint_path, data_dir):
    """Classify all images in a folder"""
    # Get class names
    class_names = get_class_names(data_dir)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        checkpoint_path,
        num_classes=len(class_names),
        model_name='convnext_tiny_in22k'
    )
    
    # Create predictor
    predictor = HairPredictor(model, device, classes=class_names)
    
    # Get all images
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]
    
    if not images:
        print(f"No images found in {folder_path}")
        return
    
    print(f"\nFound {len(images)} images. Classifying...")
    print(f"{'='*80}")
    
    # Classify each image
    results = []
    for img_path in images:
        result = predictor.predict(img_path, data_dir=data_dir)
        results.append((os.path.basename(img_path), result))
        
        print(f"{os.path.basename(img_path):40} → {result['class']:15} ({result['confidence']:.1%})")
    
    print(f"{'='*80}")
    print(f"\n✓ Classified {len(images)} images")
    
    # Summary statistics
    color_counts = {}
    for _, result in results:
        color = result['class']
        color_counts[color] = color_counts.get(color, 0) + 1
    
    print(f"\nSummary:")
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {color}: {count} images ({100*count/len(images):.1f}%)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Hair Color Classification with ConvNeXt')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data_hair_color',
                       help='Data directory to infer classes from (default: ./data_hair_color)')
    
    args = parser.parse_args()
    
    # Check that checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Check that data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print(f"Please ensure the data directory exists to infer class names.")
        return
    
    # Validate input
    if args.image and args.folder:
        print("Error: Please specify either --image or --folder, not both")
        return
    
    if not args.image and not args.folder:
        print("Error: Please specify either --image or --folder")
        return
    
    # Classify
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        classify_single_image(args.image, args.checkpoint, args.data_dir)
    else:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        classify_folder(args.folder, args.checkpoint, args.data_dir)

if __name__ == '__main__':
    main()
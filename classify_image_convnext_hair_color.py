"""
Hair Color Classification with ConvNeXt-Tiny
Classifies hair color (black/brown/blonde/etc.)

Model paths are read from config.yaml by default, but can be overridden via command-line arguments.

Usage:
    # Simple usage (reads from config.yaml)
    python classify_image_convnext_hair_color.py --image path/to/image.jpg
    python classify_image_convnext_hair_color.py --folder path/to/folder/
    
    # Override config with command-line arguments
    python classify_image_convnext_hair_color.py \
        --image path/to/image.jpg \
        --checkpoint checkpoints_hair_color/best_model.pth \
        --data-dir ./data_hair_color
"""

import argparse
import torch
import os
import yaml
from src.model import load_model
from src.predict import HairPredictor
from src.dataset import get_class_names

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("Warning: config.yaml not found. You must provide all arguments via command line.")
        return None

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
    # Load config first to get defaults
    config = load_config()
    
    # Set defaults from config if available
    if config and 'hair_color_classification' in config:
        default_checkpoint = config['hair_color_classification'].get('checkpoint', None)
        default_data_dir = config['hair_color_classification'].get('data_dir', './data_hair_color')
    else:
        default_checkpoint = None
        default_data_dir = './data_hair_color'
    
    parser = argparse.ArgumentParser(
        description='Hair Color Classification with ConvNeXt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage (reads from config.yaml)
  python classify_image_convnext_hair_color.py --image test.jpg
  python classify_image_convnext_hair_color.py --folder test_images/
  
  # Override config values
  python classify_image_convnext_hair_color.py --image test.jpg \\
      --checkpoint checkpoints_hair_color/best_model.pth \\
      --data-dir ./data_hair_color

Configuration:
  Model paths are read from config.yaml by default.
  Edit the 'hair_color_classification' section in config.yaml to set default paths.
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    
    # Model checkpoint (optional if set in config)
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint,
                       help=f'Path to hair color model checkpoint (default from config: {default_checkpoint})')
    
    # Data directory (optional if set in config)
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                       help=f'Data directory for hair color classes (default: {default_data_dir})')
    
    args = parser.parse_args()
    
    # Validate input
    if args.image and args.folder:
        print("Error: Please specify either --image or --folder, not both")
        return
    
    if not args.image and not args.folder:
        print("Error: Please specify either --image or --folder")
        return
    
    # Check that checkpoint is specified
    if not args.checkpoint:
        print("Error: Hair color checkpoint not specified. Either:")
        print("  1. Set 'hair_color_classification.checkpoint' in config.yaml, or")
        print("  2. Provide --checkpoint argument")
        return
    
    # Check that checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Hair color checkpoint not found: {args.checkpoint}")
        return
    
    # Check that data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Hair color data directory not found: {args.data_dir}")
        return
    
    # Display configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    
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
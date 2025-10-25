"""
Combined Hair Type & Color Classification with ConvNeXt-Tiny
Classifies both hair type (straight/wavy/curly) and color (black/brown/blonde/etc.) simultaneously

Model paths are read from config.yaml by default, but can be overridden via command-line arguments.

Usage:
    # Simple usage (reads from config.yaml)
    python classify_image_convnext.py --image path/to/image.jpg
    python classify_image_convnext.py --folder path/to/folder/
    
    # Override config with command-line arguments
    python classify_image_convnext.py \
        --image path/to/image.jpg \
        --checkpoint-type checkpoints_hair_type/best_model.pth \
        --checkpoint-color checkpoints_hair_color/best_model.pth \
        --data-dir-type ./data_hair_type \
        --data-dir-color ./data_hair_color
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

def load_models(checkpoint_type, checkpoint_color, data_dir_type, data_dir_color, device):
    """Load both hair type and color models"""
    
    print("Loading models...")
    
    # Get class names for both tasks
    type_classes = get_class_names(data_dir_type)
    color_classes = get_class_names(data_dir_color)
    
    print(f"  Hair Type Classes: {type_classes}")
    print(f"  Hair Color Classes: {color_classes}")
    
    # Load hair type model
    model_type = load_model(
        checkpoint_type,
        num_classes=len(type_classes),
        model_name='convnext_tiny_in22k'
    )
    predictor_type = HairPredictor(model_type, device, classes=type_classes)
    
    # Load hair color model
    model_color = load_model(
        checkpoint_color,
        num_classes=len(color_classes),
        model_name='convnext_tiny_in22k'
    )
    predictor_color = HairPredictor(model_color, device, classes=color_classes)
    
    print("✓ Models loaded successfully\n")
    
    return predictor_type, predictor_color, data_dir_type, data_dir_color

def classify_single_image(image_path, predictor_type, predictor_color, data_dir_type, data_dir_color):
    """Classify a single image for both type and color"""
    
    # Predict type
    result_type = predictor_type.predict(image_path, data_dir=data_dir_type)
    
    # Predict color
    result_color = predictor_color.predict(image_path, data_dir=data_dir_color)
    
    # Print combined results
    print(f"\n{'='*70}")
    print(f"IMAGE: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    print(f"\n🧵 HAIR TYPE")
    print(f"  Predicted: {result_type['class']}")
    print(f"  Confidence: {result_type['confidence']:.2%}")
    print(f"  Probabilities:")
    for class_name, prob in sorted(result_type['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 30)
        print(f"    {class_name:12} {prob:6.2%} {bar}")
    
    print(f"\n🎨 HAIR COLOR")
    print(f"  Predicted: {result_color['class']}")
    print(f"  Confidence: {result_color['confidence']:.2%}")
    print(f"  Probabilities:")
    for class_name, prob in sorted(result_color['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 30)
        print(f"    {class_name:12} {prob:6.2%} {bar}")
    
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY: {result_type['class'].upper()} + {result_color['class'].upper()}")
    print(f"{'='*70}\n")
    
    return result_type, result_color

def classify_folder(folder_path, predictor_type, predictor_color, data_dir_type, data_dir_color):
    """Classify all images in a folder"""
    
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
    print(f"{'='*90}")
    print(f"{'Image':<35} {'Hair Type':<20} {'Hair Color':<20} {'Combined Conf.'}")
    print(f"{'='*90}")
    
    # Classify each image
    results = []
    for img_path in images:
        result_type = predictor_type.predict(img_path, data_dir=data_dir_type)
        result_color = predictor_color.predict(img_path, data_dir=data_dir_color)
        
        results.append((os.path.basename(img_path), result_type, result_color))
        
        # Combined confidence (average)
        combined_conf = (result_type['confidence'] + result_color['confidence']) / 2
        
        print(f"{os.path.basename(img_path):<35} "
              f"{result_type['class']:<20} "
              f"{result_color['class']:<20} "
              f"{combined_conf:.1%}")
    
    print(f"{'='*90}")
    print(f"\n✓ Classified {len(images)} images")
    
    # Summary statistics
    print(f"\n{'='*90}")
    print("SUMMARY STATISTICS")
    print(f"{'='*90}")
    
    # Hair type distribution
    type_counts = {}
    for _, result_type, _ in results:
        hair_type = result_type['class']
        type_counts[hair_type] = type_counts.get(hair_type, 0) + 1
    
    print(f"\n🧵 Hair Type Distribution:")
    for hair_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(count * 40 / len(images))
        print(f"  {hair_type:12} {count:3d} images ({100*count/len(images):5.1f}%) {bar}")
    
    # Hair color distribution
    color_counts = {}
    for _, _, result_color in results:
        hair_color = result_color['class']
        color_counts[hair_color] = color_counts.get(hair_color, 0) + 1
    
    print(f"\n🎨 Hair Color Distribution:")
    for hair_color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(count * 40 / len(images))
        print(f"  {hair_color:12} {count:3d} images ({100*count/len(images):5.1f}%) {bar}")
    
    # Combined distribution (type + color)
    combined_counts = {}
    for _, result_type, result_color in results:
        combined = f"{result_type['class']} + {result_color['class']}"
        combined_counts[combined] = combined_counts.get(combined, 0) + 1
    
    print(f"\n📊 Combined Distribution (Type + Color):")
    for combined, count in sorted(combined_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {combined:25} {count:3d} images ({100*count/len(images):5.1f}%)")
    
    print(f"{'='*90}\n")
    
    return results

def main():
    # Load config first to get defaults
    config = load_config()
    
    # Set defaults from config if available
    if config and 'classification' in config:
        default_checkpoint_type = config['classification'].get('checkpoint_type', None)
        default_checkpoint_color = config['classification'].get('checkpoint_color', None)
        default_data_dir_type = config['classification'].get('data_dir_type', './data_hair_type')
        default_data_dir_color = config['classification'].get('data_dir_color', './data_hair_color')
    else:
        default_checkpoint_type = None
        default_checkpoint_color = None
        default_data_dir_type = './data_hair_type'
        default_data_dir_color = './data_hair_color'
    
    parser = argparse.ArgumentParser(
        description='Combined Hair Type & Color Classification with ConvNeXt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage (reads from config.yaml)
  python classify_image_convnext.py --image test.jpg
  python classify_image_convnext.py --folder test_images/
  
  # Override config values
  python classify_image_convnext.py --image test.jpg \\
      --checkpoint-type checkpoints_hair_type/best_model.pth \\
      --checkpoint-color checkpoints_hair_color/best_model.pth

Configuration:
  Model paths are read from config.yaml by default.
  Edit the 'classification' section in config.yaml to set default paths.
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    
    # Model checkpoints (optional if set in config)
    parser.add_argument('--checkpoint-type', type=str, default=default_checkpoint_type,
                       help=f'Path to hair type model checkpoint (default from config: {default_checkpoint_type})')
    parser.add_argument('--checkpoint-color', type=str, default=default_checkpoint_color,
                       help=f'Path to hair color model checkpoint (default from config: {default_checkpoint_color})')
    
    # Data directories (optional if set in config)
    parser.add_argument('--data-dir-type', type=str, default=default_data_dir_type,
                       help=f'Data directory for hair type classes (default: {default_data_dir_type})')
    parser.add_argument('--data-dir-color', type=str, default=default_data_dir_color,
                       help=f'Data directory for hair color classes (default: {default_data_dir_color})')
    
    args = parser.parse_args()
    
    # Validate input
    if args.image and args.folder:
        print("Error: Please specify either --image or --folder, not both")
        return
    
    if not args.image and not args.folder:
        print("Error: Please specify either --image or --folder")
        return
    
    # Check that checkpoints are specified
    if not args.checkpoint_type:
        print("Error: Hair type checkpoint not specified. Either:")
        print("  1. Set 'classification.checkpoint_type' in config.yaml, or")
        print("  2. Provide --checkpoint-type argument")
        return
    
    if not args.checkpoint_color:
        print("Error: Hair color checkpoint not specified. Either:")
        print("  1. Set 'classification.checkpoint_color' in config.yaml, or")
        print("  2. Provide --checkpoint-color argument")
        return
    
    # Check that checkpoints exist
    if not os.path.exists(args.checkpoint_type):
        print(f"Error: Hair type checkpoint not found: {args.checkpoint_type}")
        return
    
    if not os.path.exists(args.checkpoint_color):
        print(f"Error: Hair color checkpoint not found: {args.checkpoint_color}")
        return
    
    # Check that data directories exist
    if not os.path.exists(args.data_dir_type):
        print(f"Error: Hair type data directory not found: {args.data_dir_type}")
        return
    
    if not os.path.exists(args.data_dir_color):
        print(f"Error: Hair color data directory not found: {args.data_dir_color}")
        return
    
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Hair type checkpoint: {args.checkpoint_type}")
    print(f"Hair color checkpoint: {args.checkpoint_color}\n")
    
    predictor_type, predictor_color, data_dir_type, data_dir_color = load_models(
        args.checkpoint_type,
        args.checkpoint_color,
        args.data_dir_type,
        args.data_dir_color,
        device
    )
    
    # Classify
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        classify_single_image(args.image, predictor_type, predictor_color, 
                            data_dir_type, data_dir_color)
    else:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        classify_folder(args.folder, predictor_type, predictor_color,
                       data_dir_type, data_dir_color)

if __name__ == '__main__':
    main()
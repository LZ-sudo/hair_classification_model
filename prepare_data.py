"""
Prepare dataset by splitting input images into train/val/test sets

Updated to accept command-line arguments instead of reading from config.yaml

Usage:
    # For hair type dataset
    python prepare_data.py --input_dir input_pictures_processed --output_dir data
    
    # For hair color dataset
    python prepare_data.py --input_dir input_pictures_color --output_dir data_color
    
    # With custom splits
    python prepare_data.py --input_dir input_pictures_color --output_dir data_color \
                           --train_split 0.7 --val_split 0.15 --test_split 0.15
"""

import argparse
from src.dataset import prepare_dataset
import os

def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset by splitting into train/val/test sets'
    )
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with class subdirectories (e.g., input_pictures_processed)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for train/val/test splits (e.g., data)')
    
    # Optional arguments for split ratios
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Training set proportion (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation set proportion (default: 0.15)')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test set proportion (default: 0.15)')
    
    # Random seed
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Force overwrite
    parser.add_argument('--force', action='store_true',
                       help='Force overwrite if output directory exists')
    
    args = parser.parse_args()
    
    # Validate split ratios
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 1e-6:
        print(f"Error: Split ratios must sum to 1.0 (current sum: {total})")
        return
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    # Check if output directory already exists
    if os.path.exists(args.output_dir) and not args.force:
        response = input(f"\nWarning: Output directory '{args.output_dir}' already exists.\n"
                        f"This will overwrite existing splits. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Display configuration
    print("\n" + "="*70)
    print("DATASET PREPARATION CONFIGURATION")
    print("="*70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train split:      {args.train_split:.1%}")
    print(f"Val split:        {args.val_split:.1%}")
    print(f"Test split:       {args.test_split:.1%}")
    print(f"Random seed:      {args.random_seed}")
    print("="*70 + "\n")
    
    # Prepare dataset
    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        random_seed=args.random_seed
    )

if __name__ == '__main__':
    main()
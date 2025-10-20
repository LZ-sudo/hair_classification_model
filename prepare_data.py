"""
Prepare dataset by splitting input images into train/val/test sets

Run this script BEFORE training:
    python prepare_data.py
"""

import yaml
from src.dataset import prepare_dataset
import os

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    input_dir = config['input_dir']
    output_dir = config['data_dir']
    train_split = config['train_split']
    val_split = config['val_split']
    test_split = config['test_split']
    random_seed = config.get('random_seed', 42)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        print(f"Please create it with subdirectories: straight/, wavy/, curly/")
        return
    
    # Check if data directory already exists
    if os.path.exists(output_dir):
        response = input(f"\nWarning: Output directory '{output_dir}' already exists.\n"
                        f"This will overwrite existing splits. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Prepare dataset
    prepare_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        random_seed=random_seed
    )

if __name__ == '__main__':
    main()

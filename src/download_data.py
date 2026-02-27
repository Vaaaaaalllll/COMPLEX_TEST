# WARNING: template code, may need edits
"""Helper script to download and prepare a sample cat dataset.

This script helps users get started by downloading a small sample dataset.
For production use, users should prepare their own larger dataset.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import shutil


def download_sample_data():
    """Download a small sample dataset for testing.
    
    Note: This is a minimal dataset for demonstration.
    For real training, you need a much larger dataset.
    """
    print("Sample Data Preparation")
    print("=" * 60)
    print("\nThis script will help you set up the data directory structure.")
    print("You need to provide your own cat images.")
    print("\nExpected structure:")
    print("data/")
    print("  train/")
    print("    cat/        <- Put cat images here")
    print("    not_cat/    <- Put non-cat images here")
    print("  test/")
    print("    cat/        <- Put test cat images here")
    print("    not_cat/    <- Put test non-cat images here")
    
    # Create directory structure
    directories = [
        'data/train/cat',
        'data/train/not_cat',
        'data/test/cat',
        'data/test/not_cat'
    ]
    
    print("\nCreating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  7 Created {directory}")
    
    print("\n" + "=" * 60)
    print("Directory structure created successfully!")
    print("\nNext steps:")
    print("1. Add cat images to data/train/cat/ and data/test/cat/")
    print("2. Add non-cat images to data/train/not_cat/ and data/test/not_cat/")
    print("3. Recommended: At least 500+ images per class for training")
    print("4. Recommended: At least 100+ images per class for testing")
    print("\nDataset suggestions:")
    print("- Kaggle: Dogs vs Cats dataset")
    print("- Google Images (with proper licensing)")
    print("- Your own collected images")
    print("\nOnce you have images in place, run:")
    print("  python src/train.py")


def validate_dataset():
    """Validate that the dataset structure is correct."""
    required_dirs = [
        'data/train/cat',
        'data/train/not_cat',
        'data/test/cat',
        'data/test/not_cat'
    ]
    
    print("\nValidating dataset structure...")
    all_valid = True
    
    for directory in required_dirs:
        path = Path(directory)
        if not path.exists():
            print(f"  7 Missing: {directory}")
            all_valid = False
        else:
            # Count images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = [f for f in path.iterdir() 
                     if f.suffix.lower() in image_extensions]
            print(f"  7 {directory}: {len(images)} images")
            
            if len(images) == 0:
                print(f"    7 Warning: No images found in {directory}")
                all_valid = False
    
    if all_valid:
        print("\n7 Dataset structure is valid!")
    else:
        print("\n7 Please fix the issues above before training.")
    
    return all_valid


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing dataset structure')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset()
    else:
        download_sample_data()
        print("\nRun with --validate to check your dataset:")
        print("  python src/download_data.py --validate")


if __name__ == "__main__":
    main()

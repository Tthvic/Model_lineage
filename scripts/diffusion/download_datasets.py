#!/usr/bin/env python3
"""
Download COCO dataset for diffusion model lineage detection.

This script downloads the COCO 2014 training images and annotations needed for
extracting U-Net features from Stable Diffusion models.

Usage:
    python scripts/diffusion/download_datasets.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# Dataset URLs
COCO_URLS = {
    'train2014_images': 'http://images.cocodataset.org/zips/train2014.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
}

# Target directory
DATA_DIR = Path("data/datasets/coco")


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with progress bar"""
    print(f"\nDownloading: {url}")
    print(f"To: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress bar"""
    print(f"\nExtracting: {zip_path.name}")
    print(f"To: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)


def download_coco_dataset():
    """Download and extract COCO 2014 dataset"""
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COCO Dataset Download for Diffusion Model Lineage Detection")
    print("="*80)
    print(f"\nDataset will be saved to: {DATA_DIR.absolute()}")
    print(f"\nRequired downloads:")
    print(f"  1. Training images (train2014): ~13GB")
    print(f"  2. Annotations: ~241MB")
    print(f"\nTotal download size: ~13.2GB")
    print(f"This may take a while depending on your internet connection.")
    
    # Ask for confirmation
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download and extract training images
    images_zip = DATA_DIR / "train2014.zip"
    images_dir = DATA_DIR / "train2014"
    
    if not images_dir.exists():
        if not images_zip.exists():
            print("\n" + "="*80)
            print("Step 1/2: Downloading Training Images")
            print("="*80)
            download_url(COCO_URLS['train2014_images'], images_zip)
        else:
            print(f"\nTraining images zip already exists: {images_zip}")
        
        print("\n" + "="*80)
        print("Extracting Training Images")
        print("="*80)
        extract_zip(images_zip, DATA_DIR)
        
        # Clean up zip file
        print(f"\nRemoving zip file: {images_zip}")
        images_zip.unlink()
    else:
        print(f"\nTraining images already exist: {images_dir}")
    
    # Download and extract annotations
    ann_zip = DATA_DIR / "annotations_trainval2014.zip"
    ann_dir = DATA_DIR / "annotations"
    
    if not ann_dir.exists() or not (ann_dir / "captions_train2014.json").exists():
        if not ann_zip.exists():
            print("\n" + "="*80)
            print("Step 2/2: Downloading Annotations")
            print("="*80)
            download_url(COCO_URLS['annotations'], ann_zip)
        else:
            print(f"\nAnnotations zip already exists: {ann_zip}")
        
        print("\n" + "="*80)
        print("Extracting Annotations")
        print("="*80)
        extract_zip(ann_zip, DATA_DIR)
        
        # Clean up zip file
        print(f"\nRemoving zip file: {ann_zip}")
        ann_zip.unlink()
    else:
        print(f"\nAnnotations already exist: {ann_dir}")
    
    # Verify files
    print("\n" + "="*80)
    print("Verification")
    print("="*80)
    
    required_files = [
        images_dir,
        ann_dir / "captions_train2014.json",
        ann_dir / "instances_train2014.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n" + "="*80)
        print("SUCCESS: COCO dataset downloaded and extracted successfully!")
        print("="*80)
        print(f"\nYou can now run: python scripts/diffusion/generate_embeddings.py")
    else:
        print("\n" + "="*80)
        print("ERROR: Some files are missing. Please try running the script again.")
        print("="*80)
        sys.exit(1)


def main():
    """Main function"""
    try:
        download_coco_dataset()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

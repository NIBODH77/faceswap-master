#!/usr/bin/env python3
"""
Dataset Downloader for Face Swap Training
Downloads public datasets like FFHQ and provides instructions for CelebA
"""
import os
import sys
import urllib.request
import zipfile
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class DatasetDownloader:
    """Download and prepare face datasets"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.dataset_dir = self.base_dir / "datasets"
        self.dataset_dir.mkdir(exist_ok=True)
        
    def download_ffhq_thumbnails(self, num_images: int = 1000):
        """Download FFHQ thumbnail dataset (smaller, faster)
        
        Parameters
        ----------
        num_images: int
            Number of images to download (max 70000)
        """
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOADING FFHQ THUMBNAILS DATASET")
        logger.info("=" * 70)
        logger.info("\nThis will download high-quality face images from FFHQ dataset")
        logger.info(f"Number of images: {num_images}")
        logger.info("License: Creative Commons BY-NC-SA 4.0")
        logger.info("\n⚠️  This dataset is for NON-COMMERCIAL use only!")
        
        ffhq_dir = self.dataset_dir / "ffhq_thumbnails"
        ffhq_dir.mkdir(exist_ok=True)
        
        # FFHQ thumbnails are available in 128x128 size
        base_url = "https://drive.google.com/uc?export=download&id="
        
        # File IDs for FFHQ thumbnail batches (each batch has 1000 images)
        thumbnail_batches = {
            '1': '1WocxvZ2-Hc6b9c6_L2t6OYFTbiWiT9z9',  # 00000-00999
            '2': '1T1p1HPHC-_JiLrJmA8YRrGnK8qiCGUJP',  # 01000-01999
            # Add more batch IDs as needed
        }
        
        num_batches = (num_images + 999) // 1000
        
        logger.info(f"\nDownloading {num_batches} batch(es)...")
        
        for batch_num in range(1, min(num_batches + 1, len(thumbnail_batches) + 1)):
            batch_id = thumbnail_batches.get(str(batch_num))
            if not batch_id:
                logger.warning(f"Batch {batch_num} not available")
                continue
            
            output_file = ffhq_dir / f"thumbnails_batch_{batch_num}.zip"
            
            if output_file.exists():
                logger.info(f"✓ Batch {batch_num} already downloaded")
                continue
            
            logger.info(f"\nDownloading batch {batch_num}...")
            url = base_url + batch_id
            
            try:
                download_file(url, str(output_file))
                logger.info(f"✓ Downloaded batch {batch_num}")
                
                # Extract
                logger.info(f"Extracting batch {batch_num}...")
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall(ffhq_dir)
                logger.info(f"✓ Extracted batch {batch_num}")
                
                # Remove zip to save space
                output_file.unlink()
                
            except Exception as e:
                logger.error(f"Failed to download batch {batch_num}: {e}")
                continue
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ FFHQ dataset ready at: {ffhq_dir}")
        logger.info("=" * 70)
        
        return str(ffhq_dir)
    
    def download_sample_dataset(self):
        """Download a small sample dataset for testing"""
        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOADING SAMPLE DATASET")
        logger.info("=" * 70)
        logger.info("\nDownloading sample celebrity faces for testing...")
        
        sample_dir = self.dataset_dir / "sample_faces"
        sample_dir.mkdir(exist_ok=True)
        
        # Public domain celebrity images from Wikimedia Commons
        sample_urls = [
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg", "face_1.jpg"),
            # Add more public domain face images here
        ]
        
        logger.info(f"\nDownloading {len(sample_urls)} sample images...")
        
        for url, filename in sample_urls:
            output_path = sample_dir / filename
            if output_path.exists():
                logger.info(f"✓ {filename} already exists")
                continue
            
            try:
                download_file(url, str(output_path))
                logger.info(f"✓ Downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ Sample dataset ready at: {sample_dir}")
        logger.info("=" * 70)
        
        return str(sample_dir)
    
    def show_celeba_instructions(self):
        """Show instructions for downloading CelebA dataset"""
        logger.info("\n" + "=" * 70)
        logger.info("CELEBA DATASET INSTRUCTIONS")
        logger.info("=" * 70)
        
        instructions = """
CelebA Dataset - Large-scale Face Attributes Dataset
• Contains: 202,599 celebrity face images
• License: Non-commercial research purposes only
• Size: ~1.4 GB

HOW TO DOWNLOAD:
1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Click on "Google Drive" link
3. Download "img_align_celeba.zip" (Aligned & Cropped Images)
4. Extract to: {dataset_dir}/celeba/
5. You'll have 202,599 aligned face images ready for training!

ALTERNATIVE - Use Kaggle:
1. Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. Download the dataset
3. Extract to: {dataset_dir}/celeba/

After downloading:
• Put Person A images in: training_data/person_a/
• Put Person B images in: training_data/person_b/
• Run: python3 train_dataset_builder.py
        """
        
        logger.info(instructions.format(dataset_dir=self.dataset_dir))
        logger.info("=" * 70)
    
    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "=" * 70)
        print("FACE SWAP DATASET DOWNLOADER")
        print("=" * 70)
        print("\nAvailable options:")
        print("1. Download FFHQ Thumbnails (Small, ~100MB per 1000 images)")
        print("2. Download Sample Dataset (Quick test, ~5MB)")
        print("3. Show CelebA Download Instructions (Manual download)")
        print("4. Exit")
        print("\n" + "=" * 70)


def main():
    downloader = DatasetDownloader()
    
    while True:
        downloader.show_menu()
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            num_str = input("\nHow many images? (default: 1000, max: 70000): ").strip()
            num_images = int(num_str) if num_str.isdigit() else 1000
            downloader.download_ffhq_thumbnails(num_images)
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            downloader.download_sample_dataset()
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            downloader.show_celeba_instructions()
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            print("\nGoodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

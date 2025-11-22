
#!/usr/bin/env python3
"""
Ethical Dataset Builder and Training Pipeline
Uses public datasets or user's own images with proper consent
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EthicalDatasetBuilder:
    """Build training dataset ethically"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.dataset_dir = self.base_dir / "training_data"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        self.dataset_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Person A and B directories
        self.person_a_dir = self.dataset_dir / "person_a"
        self.person_b_dir = self.dataset_dir / "person_b"
        self.person_a_dir.mkdir(exist_ok=True)
        self.person_b_dir.mkdir(exist_ok=True)
    
    def use_public_dataset(self):
        """
        Use publicly available face datasets
        Example: CelebA, FFHQ (with proper attribution)
        """
        logger.info("=" * 60)
        logger.info("IMPORTANT: उचित Dataset का उपयोग करें")
        logger.info("=" * 60)
        logger.info("\nPublic Datasets जो आप उपयोग कर सकते हैं:")
        logger.info("1. CelebA - http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        logger.info("2. FFHQ - https://github.com/NVlabs/ffhq-dataset")
        logger.info("3. Your own photos with consent")
        logger.info("\nकृपया अपनी images को निम्नलिखित folders में रखें:")
        logger.info(f"Person A: {self.person_a_dir}")
        logger.info(f"Person B: {self.person_b_dir}")
        logger.info("\nप्रत्येक folder में कम से कम 500-1000 images होनी चाहिए।")
        
        return False
    
    def extract_faces(self, input_dir, output_dir):
        """Extract faces from images/videos"""
        logger.info(f"Extracting faces from {input_dir} to {output_dir}")
        
        cmd = [
            sys.executable,
            "faceswap.py",
            "extract",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-D", "s3fd",
            "-A", "fan",
            "-M", "bisenet-fp",
            "-nm", "hist"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"✓ Face extraction completed for {input_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Face extraction failed: {e}")
            return False
    
    def train_model(self, iterations=5000):
        """Train the face swap model"""
        logger.info("Starting model training...")
        
        extracted_a = self.dataset_dir / "extracted_a"
        extracted_b = self.dataset_dir / "extracted_b"
        
        cmd = [
            sys.executable,
            "faceswap.py",
            "train",
            "-A", str(extracted_a),
            "-B", str(extracted_b),
            "-m", str(self.models_dir / "trained_model"),
            "-t", "original",  # Fast, lightweight model
            "-b", "16",
            "-i", str(iterations),
            "-s", "100",
            "-I", "1000"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Model training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def run_pipeline(self):
        """Run complete ethical training pipeline"""
        logger.info("\n" + "=" * 60)
        logger.info("ETHICAL FACESWAP TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Check if datasets exist
        person_a_images = list(self.person_a_dir.glob("*"))
        person_b_images = list(self.person_b_dir.glob("*"))
        
        if not person_a_images or not person_b_images:
            logger.error("\n❌ No images found!")
            logger.error(f"Please add images to:")
            logger.error(f"  - {self.person_a_dir}")
            logger.error(f"  - {self.person_b_dir}")
            logger.error("\nप्रत्येक folder में कम से कम 500-1000 images होनी चाहिए।")
            logger.error("\nकेवल वही images उपयोग करें जिनके लिए आपके पास permission है!")
            return False
        
        logger.info(f"\nFound {len(person_a_images)} images in Person A")
        logger.info(f"Found {len(person_b_images)} images in Person B")
        
        # Extract faces
        extracted_a = self.dataset_dir / "extracted_a"
        extracted_b = self.dataset_dir / "extracted_b"
        
        if not self.extract_faces(self.person_a_dir, extracted_a):
            return False
        if not self.extract_faces(self.person_b_dir, extracted_b):
            return False
        
        # Train model
        if not self.train_model(iterations=5000):
            return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"\nTrained model saved at: {self.models_dir / 'trained_model'}")
        logger.info("\nअब आप web_app.py में इस model का उपयोग कर सकते हैं।")
        
        return True


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("FACESWAP TRAINING PIPELINE")
    print("=" * 60)
    print("\nNote: Use only images you have permission to use.\n")
    
    builder = EthicalDatasetBuilder()
    builder.run_pipeline()


if __name__ == "__main__":
    main()

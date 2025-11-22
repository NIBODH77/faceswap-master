#!/usr/bin/env python3
"""
Professional InsightFace-based face swapper
Uses state-of-the-art models for high-quality face swapping
"""
import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

logger = logging.getLogger(__name__)

class InsightFaceSwapper:
    """High-quality face swapping using InsightFace"""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the swapper with professional models
        
        Parameters
        ----------
        use_gpu: bool
            Use GPU acceleration if available
        """
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.ctx_id = 0 if self.use_gpu else -1
        
        # Set providers based on GPU availability
        if self.use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("GPU detected, using CUDA acceleration")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("Using CPU mode")
        
        logger.info("Initializing InsightFace Face Swapper...")
        
        try:
            # Initialize face analyzer with buffalo_l model (best quality)
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            logger.info("✓ Face analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load buffalo_l, trying buffalo_sc: {e}")
            try:
                self.app = FaceAnalysis(name='buffalo_sc', providers=providers)
                self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
                logger.info("✓ Face analyzer initialized with buffalo_sc")
            except Exception as e2:
                logger.error(f"Failed to initialize face analyzer: {e2}")
                raise
        
        try:
            # Initialize face swapper model
            model_path_or_name = self._download_swapper_model()
            # If model_path is just a name (not an existing file), get_model will auto-download
            self.swapper = insightface.model_zoo.get_model(model_path_or_name, providers=providers)
            logger.info("✓ Face swapper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load swapper model: {e}")
            raise
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available
        
        Returns
        -------
        bool
            True if GPU is available and CUDA is working
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.info("PyTorch not installed, GPU mode unavailable")
            return False
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def _download_swapper_model(self) -> str:
        """Download and return path to swapper model
        
        Returns
        -------
        str
            Path to the model file or model name for auto-download
        """
        model_dir = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_l')
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to find existing model
        model_name = 'inswapper_128.onnx'
        model_path = os.path.join(model_dir, model_name)
        
        if os.path.exists(model_path):
            logger.info(f"Using existing swapper model: {model_path}")
            return model_path
        
        # Try to download from official source
        logger.info("Downloading face swapper model (this may take a few minutes)...")
        import urllib.request
        
        url = 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx'
        
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"✓ Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")
            logger.info("Falling back to InsightFace managed download")
            # Return just the model name - InsightFace will auto-download it
            # This triggers InsightFace's built-in model download mechanism
            return 'inswapper_128.onnx'
    
    def enhance_face(self, face_img: np.ndarray) -> np.ndarray:
        """Apply facial enhancement
        
        Parameters
        ----------
        face_img: np.ndarray
            Face image to enhance
            
        Returns
        -------
        np.ndarray
            Enhanced face
        """
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(face_img, None, 10, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.15)
        
        # Color enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def color_correct(self, source: np.ndarray, target: np.ndarray, 
                     mask: np.ndarray) -> np.ndarray:
        """Advanced color correction using masked statistics
        
        Parameters
        ----------
        source: np.ndarray
            Source image
        target: np.ndarray  
            Target image
        mask: np.ndarray
            Mask for region of interest (0-255 uint8 or 0-1 float)
            
        Returns
        -------
        np.ndarray
            Color corrected source
        """
        # Ensure mask is in proper format
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask_bool = mask > 0.3
        else:
            mask_bool = mask > 0
        
        # Check if mask has sufficient area
        if np.sum(mask_bool) < 100:
            logger.warning("Mask area too small for color correction, skipping")
            return source
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        corrected_lab = source_lab.copy().astype(np.float32)
        
        for i in range(3):
            source_channel = source_lab[:,:,i][mask_bool]
            target_channel = target_lab[:,:,i][mask_bool]
            
            # Safety check for valid data
            if len(source_channel) == 0 or len(target_channel) == 0:
                continue
            
            source_mean = source_channel.mean()
            source_std = source_channel.std()
            target_mean = target_channel.mean()
            target_std = target_channel.std()
            
            # Avoid division by zero
            if source_std < 1e-6:
                source_std = 1.0
            
            # Match statistics
            corrected_lab[:,:,i] = (corrected_lab[:,:,i] - source_mean) * (target_std / source_std) + target_mean
        
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def create_seamless_mask(self, img_shape: Tuple[int, int], 
                            bbox: np.ndarray, margin: float = 0.1) -> np.ndarray:
        """Create seamless blending mask
        
        Parameters
        ----------
        img_shape: Tuple[int, int]
            Image shape (height, width)
        bbox: np.ndarray
            Face bounding box
        margin: float
            Margin around face for blending
            
        Returns
        -------
        np.ndarray
            Blending mask
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Extract bbox
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add margin
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1_m = max(0, x1 - margin_x)
        y1_m = max(0, y1 - margin_y)
        x2_m = min(w, x2 + margin_x)
        y2_m = min(h, y2 + margin_y)
        
        # Create elliptical mask
        center_x = (x1_m + x2_m) // 2
        center_y = (y1_m + y2_m) // 2
        radius_x = (x2_m - x1_m) // 2
        radius_y = (y2_m - y1_m) // 2
        
        Y, X = np.ogrid[:h, :w]
        dist = ((X - center_x) / radius_x) ** 2 + ((Y - center_y) / radius_y) ** 2
        mask = np.clip(1 - dist, 0, 1)
        
        # Apply Gaussian blur for smooth blending
        mask = cv2.GaussianBlur(mask, (99, 99), 30)
        
        return mask
    
    def blend_seamless(self, source: np.ndarray, target: np.ndarray, 
                      mask: np.ndarray) -> np.ndarray:
        """Seamless Poisson blending
        
        Parameters
        ----------
        source: np.ndarray
            Source image
        target: np.ndarray
            Target image
        mask: np.ndarray
            Blending mask
            
        Returns
        -------
        np.ndarray
            Blended result
        """
        # Ensure same size
        if source.shape[:2] != target.shape[:2]:
            source = cv2.resize(source, (target.shape[1], target.shape[0]))
        
        # Create binary mask for seamless clone
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Find center
        moments = cv2.moments(binary_mask)
        if moments['m00'] == 0:
            # Fallback to simple blending
            mask_3ch = np.stack([mask] * 3, axis=-1)
            return (source * mask_3ch + target * (1 - mask_3ch)).astype(np.uint8)
        
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        
        try:
            # Seamless clone
            result = cv2.seamlessClone(source, target, binary_mask, 
                                      (center_x, center_y), cv2.MIXED_CLONE)
            return result
        except:
            # Fallback to alpha blending
            mask_3ch = np.stack([mask] * 3, axis=-1)
            return (source * mask_3ch + target * (1 - mask_3ch)).astype(np.uint8)
    
    def swap_faces(self, source_path: str, target_path: str, 
                   output_path: str, blend_strength: float = 0.85) -> bool:
        """Perform high-quality face swap
        
        Parameters
        ----------
        source_path: str
            Path to source image
        target_path: str
            Path to target image  
        output_path: str
            Path to save result
        blend_strength: float
            Blending strength (0-1)
            
        Returns
        -------
        bool
            Success status
        """
        try:
            # Load images
            source_img = cv2.imread(source_path)
            target_img = cv2.imread(target_path)
            
            if source_img is None or target_img is None:
                logger.error("Failed to load images")
                return False
            
            logger.info(f"Processing images: {source_img.shape}, {target_img.shape}")
            
            # Detect faces
            source_faces = self.app.get(source_img)
            target_faces = self.app.get(target_img)
            
            if len(source_faces) == 0:
                logger.error("No face detected in source image")
                return False
            
            if len(target_faces) == 0:
                logger.error("No face detected in target image")
                return False
            
            # Use largest faces
            source_face = max(source_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            target_face = max(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            
            logger.info(f"✓ Detected faces - Source confidence: {source_face.det_score:.2f}, Target confidence: {target_face.det_score:.2f}")
            
            # Perform face swap
            result = target_img.copy()
            result = self.swapper.get(result, target_face, source_face, paste_back=True)
            
            # Post-processing for better quality
            # Create seamless blend mask
            mask = self.create_seamless_mask(result.shape[:2], target_face.bbox)
            
            # Convert mask for color correction (ensure uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Color correction
            result = self.color_correct(result, target_img, mask_uint8)
            
            # Seamless blending
            result = self.blend_seamless(result, target_img, mask * blend_strength)
            
            # Enhance final result
            face_region = target_face.bbox.astype(int)
            x1, y1, x2, y2 = face_region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(result.shape[1], x2), min(result.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                result[y1:y2, x1:x2] = self.enhance_face(result[y1:y2, x1:x2])
            
            # Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"✓ High-quality face swap completed: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Face swap failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Test the swapper
    swapper = InsightFaceSwapper(use_gpu=False)
    
    # Example usage
    success = swapper.swap_faces(
        "uploads/source.jpg",
        "uploads/target.jpg",
        "outputs/swapped_hq.jpg",
        blend_strength=0.85
    )
    
    print(f"High-quality swap {'successful' if success else 'failed'}")

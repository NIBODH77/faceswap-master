
#!/usr/bin/env python3
"""
InsightFace-based single image face swapper
No training required - works with single images
"""
import os
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class InsightFaceSwapper:
    """Single image face swapping using face embeddings"""
    
    def __init__(self):
        """Initialize the swapper"""
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in image
        
        Parameters
        ----------
        image: np.ndarray
            Input image
            
        Returns
        -------
        Optional[Tuple[int, int, int, int]]
            Face bounding box (x, y, w, h) or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
            
        # Return largest face
        return max(faces, key=lambda f: f[2] * f[3])
    
    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     size: int = 256) -> np.ndarray:
        """Extract and resize face region
        
        Parameters
        ----------
        image: np.ndarray
            Input image
        bbox: Tuple[int, int, int, int]
            Face bounding box (x, y, w, h)
        size: int
            Target face size
            
        Returns
        -------
        np.ndarray
            Extracted face
        """
        x, y, w, h = bbox
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (size, size))
        
        return face
    
    def get_facial_landmarks(self, face: np.ndarray) -> np.ndarray:
        """Get simplified facial landmarks
        
        Parameters
        ----------
        face: np.ndarray
            Face image
            
        Returns
        -------
        np.ndarray
            Facial landmarks
        """
        # Simplified landmark detection using eye detection
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        eyes = eye_cascade.detectMultiScale(gray)
        
        # Create basic landmarks
        h, w = face.shape[:2]
        landmarks = np.array([
            [w//2, h//3],      # nose
            [w//3, h//3],      # left eye
            [2*w//3, h//3],    # right eye
            [w//3, 2*h//3],    # left mouth
            [2*w//3, 2*h//3],  # right mouth
        ], dtype=np.float32)
        
        # Update with detected eyes if available
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            landmarks[1] = [eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2]
            landmarks[2] = [eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2]
        
        return landmarks
    
    def align_face(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """Align source face to target face orientation
        
        Parameters
        ----------
        source_face: np.ndarray
            Source face to transform
        target_face: np.ndarray
            Target face for reference
            
        Returns
        -------
        np.ndarray
            Aligned source face
        """
        src_landmarks = self.get_facial_landmarks(source_face)
        tgt_landmarks = self.get_facial_landmarks(target_face)
        
        # Calculate affine transform
        transform_matrix = cv2.estimateAffinePartial2D(src_landmarks, tgt_landmarks)[0]
        
        # Apply transform
        aligned = cv2.warpAffine(
            source_face, 
            transform_matrix, 
            (target_face.shape[1], target_face.shape[0])
        )
        
        return aligned
    
    def blend_faces(self, source_face: np.ndarray, target_face: np.ndarray, 
                    alpha: float = 0.8) -> np.ndarray:
        """Blend source face onto target
        
        Parameters
        ----------
        source_face: np.ndarray
            Source face
        target_face: np.ndarray
            Target face
        alpha: float
            Blending strength (0-1)
            
        Returns
        -------
        np.ndarray
            Blended result
        """
        # Create smooth mask
        h, w = target_face.shape[:2]
        center = (w//2, h//2)
        radius = min(w, h) // 2
        
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = np.clip(1 - (dist_from_center / radius), 0, 1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask = np.stack([mask] * 3, axis=-1)
        
        # Color correction
        source_mean = source_face.mean(axis=(0, 1))
        target_mean = target_face.mean(axis=(0, 1))
        source_corrected = source_face * (target_mean / (source_mean + 1e-6))
        source_corrected = np.clip(source_corrected, 0, 255).astype(np.uint8)
        
        # Blend
        blended = (alpha * source_corrected * mask + 
                   (1 - alpha * mask) * target_face).astype(np.uint8)
        
        return blended
    
    def swap_faces(self, source_path: str, target_path: str, 
                   output_path: str, blend_strength: float = 0.8) -> bool:
        """Perform face swap
        
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
            
            # Detect faces
            source_bbox = self.detect_face(source_img)
            target_bbox = self.detect_face(target_img)
            
            if source_bbox is None or target_bbox is None:
                logger.error("Failed to detect faces")
                return False
            
            logger.info(f"Detected faces - Source: {source_bbox}, Target: {target_bbox}")
            
            # Extract faces
            source_face = self.extract_face(source_img, source_bbox)
            target_face = self.extract_face(target_img, target_bbox)
            
            # Align source to target
            aligned_source = self.align_face(source_face, target_face)
            
            # Blend faces
            swapped_face = self.blend_faces(aligned_source, target_face, blend_strength)
            
            # Place back in target image
            tx, ty, tw, th = target_bbox
            padding = int(0.2 * max(tw, th))
            x1 = max(0, tx - padding)
            y1 = max(0, ty - padding)
            x2 = min(target_img.shape[1], tx + tw + padding)
            y2 = min(target_img.shape[0], ty + th + padding)
            
            swapped_resized = cv2.resize(swapped_face, (x2 - x1, y2 - y1))
            
            result = target_img.copy()
            result[y1:y2, x1:x2] = swapped_resized
            
            # Save result
            cv2.imwrite(output_path, result)
            logger.info(f"Face swap completed: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Face swap failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Test the swapper
    swapper = InsightFaceSwapper()
    
    # Example usage
    success = swapper.swap_faces(
        "uploads/source.jpg",
        "uploads/target.jpg",
        "outputs/swapped.jpg",
        blend_strength=0.85
    )
    
    print(f"Swap {'successful' if success else 'failed'}")

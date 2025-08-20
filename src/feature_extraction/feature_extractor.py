"""
Feature extraction module for classical machine learning approach.
Implements comprehensive feature extraction including color, texture, and shape features.
"""

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage import color, measure
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import Config
from ..utils.logger import get_logger


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        pass


class ColorFeatureExtractor(FeatureExtractor):
    """Extract color-based features from images."""
    
    def __init__(self, hist_bins: int = 32):
        """
        Initialize color feature extractor.
        
        Args:
            hist_bins: Number of bins for histograms
        """
        self.hist_bins = hist_bins
        self.logger = get_logger(__name__)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color features from image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # RGB histograms
        for channel in range(3):
            hist = cv2.calcHist([image], [channel], None, [self.hist_bins], [0, 256])
            features.extend(hist.flatten())
        
        # HSV histograms
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [self.hist_bins], [0, 256])
            features.extend(hist.flatten())
        
        # Color moments (mean, std, skewness) for each channel
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    self._calculate_skewness(channel_data)
                ])
            else:
                features.extend([0, 0, 0])
        
        # HSV moments
        for channel in range(3):
            channel_data = hsv[:, :, channel].flatten()
            if len(channel_data) > 0:
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    self._calculate_skewness(channel_data)
                ])
            else:
                features.extend([0, 0, 0])
        
        return np.array(features)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    
    def get_feature_names(self) -> List[str]:
        """Get names of color features."""
        names = []
        
        # RGB histogram features
        for channel in ['R', 'G', 'B']:
            for i in range(self.hist_bins):
                names.append(f'hist_{channel}_{i}')
        
        # HSV histogram features
        for channel in ['H', 'S', 'V']:
            for i in range(self.hist_bins):
                names.append(f'hist_{channel}_{i}')
        
        # RGB moments
        for channel in ['R', 'G', 'B']:
            names.extend([f'{channel}_mean', f'{channel}_std', f'{channel}_skew'])
        
        # HSV moments
        for channel in ['H', 'S', 'V']:
            names.extend([f'{channel}_mean', f'{channel}_std', f'{channel}_skew'])
        
        return names


class TextureFeatureExtractor(FeatureExtractor):
    """Extract texture-based features from images."""
    
    def __init__(self, 
                 lbp_radius: int = 3, 
                 lbp_n_points: int = 24,
                 glcm_distances: List[int] = None,
                 glcm_angles: List[float] = None):
        """
        Initialize texture feature extractor.
        
        Args:
            lbp_radius: Radius for LBP
            lbp_n_points: Number of points for LBP
            glcm_distances: Distances for GLCM
            glcm_angles: Angles for GLCM
        """
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.glcm_distances = glcm_distances or [5]
        self.glcm_angles = glcm_angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.logger = get_logger(__name__)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features from image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector as numpy array
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = []
        
        # Local Binary Pattern (LBP)
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # Gray Level Co-occurrence Matrix (GLCM)
        glcm_features = self._extract_glcm_features(gray)
        features.extend(glcm_features)
        
        return np.array(features)
    
    def _extract_lbp_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract LBP features."""
        try:
            lbp = local_binary_pattern(gray_image, self.lbp_n_points, 
                                     self.lbp_radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_n_points + 2, 
                                     range=(0, self.lbp_n_points + 2))
            # Normalize histogram
            lbp_hist = lbp_hist.astype(float)
            if np.sum(lbp_hist) > 0:
                lbp_hist = lbp_hist / np.sum(lbp_hist)
            
            return lbp_hist.tolist()
            
        except Exception as e:
            self.logger.warning(f"LBP extraction failed: {e}")
            return [0.0] * (self.lbp_n_points + 2)
    
    def _extract_glcm_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract GLCM features."""
        try:
            # Downsample for efficiency
            if gray_image.shape[0] > 64 or gray_image.shape[1] > 64:
                gray_small = cv2.resize(gray_image, (64, 64))
            else:
                gray_small = gray_image
            
            # Compute GLCM
            glcm = graycomatrix(gray_small, self.glcm_distances, self.glcm_angles, 
                              256, symmetric=True, normed=True)
            
            features = []
            # Extract GLCM properties
            for prop in ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'energy']:
                prop_values = graycoprops(glcm, prop)
                features.extend(prop_values.flatten().tolist())
            
            return features
            
        except Exception as e:
            self.logger.warning(f"GLCM extraction failed: {e}")
            # Return zeros for all GLCM features
            n_features = len(self.glcm_distances) * len(self.glcm_angles) * 5  # 5 properties
            return [0.0] * n_features
    
    def get_feature_names(self) -> List[str]:
        """Get names of texture features."""
        names = []
        
        # LBP features
        for i in range(self.lbp_n_points + 2):
            names.append(f'lbp_bin_{i}')
        
        # GLCM features
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'energy']
        for prop in properties:
            for d_idx, distance in enumerate(self.glcm_distances):
                for a_idx, angle in enumerate(self.glcm_angles):
                    angle_deg = int(np.degrees(angle))
                    names.append(f'glcm_{prop}_d{distance}_a{angle_deg}')
        
        return names


class ShapeFeatureExtractor(FeatureExtractor):
    """Extract shape and geometric features from images."""
    
    def __init__(self):
        """Initialize shape feature extractor."""
        self.logger = get_logger(__name__)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract shape features from image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector as numpy array
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        try:
            # Binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                features = self._extract_contour_features(largest_contour, gray.shape)
            else:
                # No contours found
                features = [0.0] * 8  # Number of shape features
            
            return np.array(features)
            
        except Exception as e:
            self.logger.warning(f"Shape extraction failed: {e}")
            return np.array([0.0] * 8)
    
    def _extract_contour_features(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> List[float]:
        """Extract features from a contour."""
        features = []
        
        # Basic geometric properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Derived features
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h) if w * h > 0 else 0
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Compactness (perimeter^2 / area)
        compactness = (perimeter * perimeter) / area if area > 0 else 0
        
        # Relative area (area / image area)
        image_area = image_shape[0] * image_shape[1]
        relative_area = area / image_area if image_area > 0 else 0
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
        
        features = [
            area,
            perimeter, 
            aspect_ratio,
            extent,
            solidity,
            compactness,
            relative_area,
            equivalent_diameter
        ]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of shape features."""
        return [
            'area',
            'perimeter',
            'aspect_ratio', 
            'extent',
            'solidity',
            'compactness',
            'relative_area',
            'equivalent_diameter'
        ]


class HOGFeatureExtractor(FeatureExtractor):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    
    def __init__(self, 
                 orientations: int = 9,
                 pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Initialize HOG feature extractor.
        
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell
            cells_per_block: Number of cells in each block
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.logger = get_logger(__name__)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Extract HOG features
            hog_features = hog(gray,
                             orientations=self.orientations,
                             pixels_per_cell=self.pixels_per_cell,
                             cells_per_block=self.cells_per_block,
                             visualize=False,
                             transform_sqrt=True,
                             feature_vector=True)
            
            return hog_features
            
        except Exception as e:
            self.logger.warning(f"HOG extraction failed: {e}")
            # Calculate expected feature size
            h, w = image.shape[:2]
            n_cells_h = h // self.pixels_per_cell[0]
            n_cells_w = w // self.pixels_per_cell[1]
            n_blocks_h = n_cells_h - self.cells_per_block[0] + 1
            n_blocks_w = n_cells_w - self.cells_per_block[1] + 1
            n_features = (n_blocks_h * n_blocks_w * 
                         self.cells_per_block[0] * self.cells_per_block[1] * 
                         self.orientations)
            return np.zeros(max(1, n_features))
    
    def get_feature_names(self) -> List[str]:
        """Get names of HOG features."""
        # HOG features don't have meaningful individual names
        # Return generic names
        return [f'hog_feature_{i}' for i in range(1000)]  # Placeholder


class ComprehensiveFeatureExtractor:
    """Comprehensive feature extractor combining all feature types."""
    
    def __init__(self, config: Config):
        """
        Initialize comprehensive feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize individual extractors
        self.color_extractor = ColorFeatureExtractor(config.classical_ml.color_hist_bins)
        self.texture_extractor = TextureFeatureExtractor(
            lbp_radius=config.classical_ml.lbp_radius,
            lbp_n_points=config.classical_ml.lbp_n_points,
            glcm_distances=config.classical_ml.glcm_distances,
            glcm_angles=config.classical_ml.glcm_angles
        )
        self.shape_extractor = ShapeFeatureExtractor()
        self.hog_extractor = HOGFeatureExtractor()
        
        # Cache for feature names
        self._feature_names = None
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract all features from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Combined feature vector or None if extraction fails
        """
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return None
            
            # Extract features from each extractor
            color_features = self.color_extractor.extract(image)
            texture_features = self.texture_extractor.extract(image)
            shape_features = self.shape_extractor.extract(image)
            hog_features = self.hog_extractor.extract(image)
            
            # Combine all features
            all_features = np.concatenate([
                color_features,
                texture_features, 
                shape_features,
                hog_features
            ])
            
            return all_features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for feature extraction."""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and pad to target size
            img = self._resize_and_pad(img)
            
            return img
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed for {image_path}: {e}")
            return None
    
    def _resize_and_pad(self, image: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio and pad to square."""
        h, w = image.shape[:2]
        target_size = self.config.data.img_size
        
        # Calculate scale to fit within target size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_w, pad_h = target_size - new_w, target_size - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=0)
        
        return image
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        if self._feature_names is None:
            self._feature_names = []
            self._feature_names.extend(self.color_extractor.get_feature_names())
            self._feature_names.extend(self.texture_extractor.get_feature_names())
            self._feature_names.extend(self.shape_extractor.get_feature_names())
            
            # For HOG, we need to extract from a dummy image to get the exact count
            dummy_image = np.zeros((self.config.data.img_size, self.config.data.img_size, 3), dtype=np.uint8)
            hog_features = self.hog_extractor.extract(dummy_image)
            hog_names = [f'hog_feature_{i}' for i in range(len(hog_features))]
            self._feature_names.extend(hog_names)
        
        return self._feature_names
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about feature extraction."""
        dummy_image = np.zeros((self.config.data.img_size, self.config.data.img_size, 3), dtype=np.uint8)
        
        color_features = self.color_extractor.extract(dummy_image)
        texture_features = self.texture_extractor.extract(dummy_image)
        shape_features = self.shape_extractor.extract(dummy_image)
        hog_features = self.hog_extractor.extract(dummy_image)
        
        info = {
            'total_features': len(color_features) + len(texture_features) + len(shape_features) + len(hog_features),
            'color_features': len(color_features),
            'texture_features': len(texture_features),
            'shape_features': len(shape_features),
            'hog_features': len(hog_features),
            'feature_breakdown': {
                'color': {
                    'count': len(color_features),
                    'description': 'RGB/HSV histograms and color moments'
                },
                'texture': {
                    'count': len(texture_features),
                    'description': 'LBP and GLCM texture descriptors'
                },
                'shape': {
                    'count': len(shape_features),
                    'description': 'Geometric and morphological features'
                },
                'hog': {
                    'count': len(hog_features),
                    'description': 'Histogram of Oriented Gradients'
                }
            }
        }
        
        return info


# Utility function for batch feature extraction
def extract_features_batch(image_paths: List[str], 
                          config: Config,
                          show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from multiple images.
    
    Args:
        image_paths: List of image file paths
        config: Configuration object
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (features_array, successful_paths)
    """
    from tqdm import tqdm
    
    extractor = ComprehensiveFeatureExtractor(config)
    
    features_list = []
    successful_paths = []
    
    iterator = tqdm(image_paths, desc="Extracting features") if show_progress else image_paths
    
    for img_path in iterator:
        features = extractor.extract_features(img_path)
        if features is not None:
            features_list.append(features)
            successful_paths.append(img_path)
    
    if features_list:
        features_array = np.array(features_list)
        return features_array, successful_paths
    else:
        return np.array([]), []
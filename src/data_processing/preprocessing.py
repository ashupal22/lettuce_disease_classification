"""
Data preprocessing module for lettuce disease classification.
Handles image preprocessing, data augmentation, and dataset preparation.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, datasets
from torchvision.transforms import functional as TF
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional
from abc import ABC, abstractmethod
import pickle

from ..utils.config import Config
from ..utils.logger import get_logger


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing."""
    
    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess a single image."""
        pass


class ResizeAndPadPreprocessor(ImagePreprocessor):
    """Resize and pad preprocessor that maintains aspect ratio."""
    
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio and pad to square.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        image = TF.resize(image, (new_h, new_w))
        
        # Pad to square
        pad_w, pad_h = self.target_size - new_w, self.target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        image = TF.pad(image, padding, fill=0)  # Black padding
        
        return image


class ResizePadTransform:
    """Transform wrapper for ResizeAndPadPreprocessor."""
    
    def __init__(self, size: int = 224):
        self.preprocessor = ResizeAndPadPreprocessor(size)
    
    def __call__(self, img):
        return self.preprocessor.preprocess(img)


class DataAugmentationFactory:
    """Factory for creating data augmentation pipelines."""
    
    @staticmethod
    def create_basic_transform(img_size: int = 224) -> transforms.Compose:
        """Create basic transformation pipeline (no augmentation)."""
        return transforms.Compose([
            ResizePadTransform(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    @staticmethod
    def create_augmented_transform(img_size: int = 224, 
                                 rotation_degrees: int = 20,
                                 horizontal_flip_prob: float = 0.5,
                                 crop_scale_min: float = 0.8,
                                 crop_scale_max: float = 1.0) -> transforms.Compose:
        """Create augmented transformation pipeline."""
        return transforms.Compose([
            ResizePadTransform(img_size),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.RandomRotation(rotation_degrees),
            transforms.RandomResizedCrop(img_size, scale=(crop_scale_min, crop_scale_max)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def create_test_transform(img_size: int = 224) -> transforms.Compose:
        """Create test transformation pipeline (deterministic)."""
        return transforms.Compose([
            ResizePadTransform(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class FilteredDataset(Dataset):
    """Filtered dataset for hierarchical classification."""
    
    def __init__(self, 
                 base_dataset: Dataset,
                 include_classes: List[int],
                 binary: bool = False,
                 target_map: Dict[int, int] = None,
                 transform: transforms.Compose = None,
                 remap_to_sequential: bool = False):
        """
        Initialize filtered dataset.
        
        Args:
            base_dataset: Base dataset to filter
            include_classes: List of class indices to include
            binary: Whether this is binary classification
            target_map: Mapping from original to new labels
            transform: Transform to apply to images
            remap_to_sequential: Whether to remap classes to sequential indices
        """
        self.base_dataset = base_dataset
        self.include_classes = include_classes
        self.binary = binary
        self.target_map = target_map
        self.transform = transform
        self.remap_to_sequential = remap_to_sequential
        
        # Filter indices
        self.indices = []
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            if label in include_classes:
                self.indices.append(i)
        
        # Create sequential mapping if needed
        if self.remap_to_sequential and not self.binary:
            self.class_to_sequential = {cls: i for i, cls in enumerate(sorted(include_classes))}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base_dataset[real_idx]
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        
        # Apply label mapping
        if self.binary and self.target_map:
            label = self.target_map[label]
        elif self.remap_to_sequential and not self.binary:
            label = self.class_to_sequential[label]
        
        return img, label


class HierarchicalDatasetBuilder:
    """Builder for creating hierarchical datasets."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Transform factories
        self.basic_transform = DataAugmentationFactory.create_basic_transform(
            self.config.data.img_size)
        self.augmented_transform = DataAugmentationFactory.create_augmented_transform(
            self.config.data.img_size,
            self.config.deep_learning.rotation_degrees,
            self.config.deep_learning.horizontal_flip_prob,
            self.config.deep_learning.crop_scale_min,
            self.config.deep_learning.crop_scale_max
        )
        self.test_transform = DataAugmentationFactory.create_test_transform(
            self.config.data.img_size)
    
    def build_datasets(self) -> Dict[str, Any]:
        """
        Build hierarchical datasets for training and evaluation.
        
        Returns:
            Dictionary containing all datasets and metadata
        """
        self.logger.info("🔄 Building hierarchical datasets...")
        
        # Load base dataset
        base_dataset = datasets.ImageFolder(
            self.config.data.raw_data_dir, 
            transform=self.basic_transform
        )
        
        # Verify class names match
        if base_dataset.classes != self.config.data.class_names:
            self.logger.warning("Dataset classes don't match config classes")
            self.logger.info(f"Dataset classes: {base_dataset.classes}")
            self.logger.info(f"Config classes: {self.config.data.class_names}")
        
        # Split into train and test
        train_size = int((1 - self.config.data.test_size) * len(base_dataset))
        test_size = len(base_dataset) - train_size
        
        train_dataset, test_dataset = random_split(
            base_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        self.logger.info(f"📊 Dataset split: {train_size} train, {test_size} test")
        
        # Build hierarchical datasets
        datasets_dict = self._build_hierarchical_splits(train_dataset, test_dataset)
        
        # Add metadata
        datasets_dict['metadata'] = {
            'total_samples': len(base_dataset),
            'train_size': train_size,
            'test_size': test_size,
            'class_names': self.config.data.class_names,
            'healthy_idx': self.config.data.healthy_idx,
            'weed_idx': self.config.data.weed_idx,
            'disease_indices': self.config.data.disease_indices
        }
        
        self.logger.info("✅ Hierarchical datasets built successfully!")
        return datasets_dict
    
    def _build_hierarchical_splits(self, train_dataset: Subset, test_dataset: Subset) -> Dict[str, Any]:
        """Build the three hierarchical classification datasets."""
        
        # Dataset 1: Crop vs Weed (binary)
        self.logger.info("🌱 Building Dataset 1: Crop vs Weed")
        
        crop_weed_map = {
            self.config.data.weed_idx: 0,  # Weed
            **{i: 1 for i in range(len(self.config.data.class_names)) if i != self.config.data.weed_idx}  # Crop
        }
        
        train_ds1 = FilteredDataset(
            train_dataset,
            include_classes=list(range(len(self.config.data.class_names))),
            binary=True,
            target_map=crop_weed_map,
            transform=self.basic_transform
        )
        
        test_ds1 = FilteredDataset(
            test_dataset,
            include_classes=list(range(len(self.config.data.class_names))),
            binary=True,
            target_map=crop_weed_map,
            transform=self.test_transform
        )
        
        # Dataset 2: Healthy vs Disease (binary, crops only)
        self.logger.info("🏥 Building Dataset 2: Healthy vs Disease")
        
        healthy_disease_map = {
            self.config.data.healthy_idx: 1,  # Healthy
            **{i: 0 for i in self.config.data.disease_indices}  # Disease
        }
        
        crop_classes = [self.config.data.healthy_idx] + self.config.data.disease_indices
        
        # For training, use augmentation for disease classes
        train_ds2_healthy = FilteredDataset(
            train_dataset,
            include_classes=[self.config.data.healthy_idx],
            binary=True,
            target_map=healthy_disease_map,
            transform=self.basic_transform
        )
        
        train_ds2_disease = FilteredDataset(
            train_dataset,
            include_classes=self.config.data.disease_indices,
            binary=True,
            target_map=healthy_disease_map,
            transform=self.augmented_transform  # Augment minority classes
        )
        
        train_ds2 = torch.utils.data.ConcatDataset([train_ds2_healthy, train_ds2_disease])
        
        test_ds2 = FilteredDataset(
            test_dataset,
            include_classes=crop_classes,
            binary=True,
            target_map=healthy_disease_map,
            transform=self.test_transform
        )
        
        # Dataset 3: Disease Classification (multiclass, diseases only)
        self.logger.info("🦠 Building Dataset 3: Disease Classification")
        
        train_ds3 = FilteredDataset(
            train_dataset,
            include_classes=self.config.data.disease_indices,
            binary=False,
            transform=self.augmented_transform,
            remap_to_sequential=True
        )
        
        test_ds3 = FilteredDataset(
            test_dataset,
            include_classes=self.config.data.disease_indices,
            binary=False,
            transform=self.test_transform,
            remap_to_sequential=True
        )
        
        # Log dataset sizes
        self.logger.info(f"📊 Dataset sizes:")
        self.logger.info(f"  - Train DS1 (Crop vs Weed): {len(train_ds1)}")
        self.logger.info(f"  - Train DS2 (Healthy vs Disease): {len(train_ds2)}")
        self.logger.info(f"  - Train DS3 (Disease Classification): {len(train_ds3)}")
        self.logger.info(f"  - Test DS1: {len(test_ds1)}")
        self.logger.info(f"  - Test DS2: {len(test_ds2)}")
        self.logger.info(f"  - Test DS3: {len(test_ds3)}")
        
        return {
            'train_ds1': train_ds1,
            'train_ds2': train_ds2,
            'train_ds3': train_ds3,
            'test_ds1': test_ds1,
            'test_ds2': test_ds2,
            'test_ds3': test_ds3,
            'test_full': test_dataset
        }
    
    def create_data_loaders(self, datasets: Dict[str, Any]) -> Dict[str, DataLoader]:
        """Create data loaders for all datasets."""
        self.logger.info("🔄 Creating data loaders...")
        
        loaders = {}
        batch_size = self.config.deep_learning.batch_size
        
        # Training loaders
        loaders['train_loader1'] = DataLoader(
            datasets['train_ds1'], 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        loaders['train_loader2'] = DataLoader(
            datasets['train_ds2'], 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        loaders['train_loader3'] = DataLoader(
            datasets['train_ds3'], 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Test loaders
        loaders['test_loader1'] = DataLoader(
            datasets['test_ds1'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        loaders['test_loader2'] = DataLoader(
            datasets['test_ds2'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        loaders['test_loader3'] = DataLoader(
            datasets['test_ds3'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        loaders['test_loader_full'] = DataLoader(
            datasets['test_full'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info("✅ Data loaders created successfully!")
        return loaders


class ClassicalMLPreprocessor:
    """Preprocessor for classical ML features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        self.preprocessor = ResizeAndPadPreprocessor(config.data.img_size)
    
    def prepare_classical_ml_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for classical ML by extracting file paths and labels.
        
        Returns:
            Tuple of (file_paths, labels) arrays
        """
        self.logger.info("📁 Preparing file paths for classical ML...")
        
        file_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.config.data.class_names):
            class_dir = os.path.join(self.config.data.raw_data_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                file_paths.append(img_path)
                labels.append(class_idx)
        
        self.logger.info(f"📊 Found {len(file_paths)} images for classical ML")
        return np.array(file_paths), np.array(labels)
    
    def preprocess_image_for_classical_ml(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image for classical ML feature extraction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array or None if error
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and pad
            h, w = img.shape[:2]
            scale = self.config.data.img_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            img = cv2.resize(img, (new_w, new_h))
            
            # Padding
            pad_w, pad_h = self.config.data.img_size - new_w, self.config.data.img_size - new_h
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            
            return img
            
        except Exception as e:
            self.logger.warning(f"Error preprocessing {image_path}: {e}")
            return None


class DatasetSplitter:
    """Utility class for dataset splitting operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
    
    def create_stratified_split(self, 
                              file_paths: np.ndarray, 
                              labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train-test split.
        
        Args:
            file_paths: Array of file paths
            labels: Array of labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            file_paths, labels,
            test_size=self.config.data.test_size,
            random_state=self.config.seed,
            stratify=labels
        )
    
    def create_hierarchical_splits(self, 
                                 X_train: np.ndarray, 
                                 X_test: np.ndarray,
                                 y_train: np.ndarray, 
                                 y_test: np.ndarray) -> Dict[str, Tuple]:
        """
        Create hierarchical splits for classical ML.
        
        Args:
            X_train, X_test: Training and test feature arrays
            y_train, y_test: Training and test label arrays
            
        Returns:
            Dictionary containing hierarchical splits
        """
        self.logger.info("🔄 Creating hierarchical splits for classical ML...")
        
        # Split 1: Crop vs Weed
        y_train_1 = np.where(y_train == self.config.data.weed_idx, 0, 1)
        y_test_1 = np.where(y_test == self.config.data.weed_idx, 0, 1)
        
        # Split 2: Healthy vs Disease (crops only)
        crop_mask_train = y_train != self.config.data.weed_idx
        crop_mask_test = y_test != self.config.data.weed_idx
        
        X_train_2 = X_train[crop_mask_train]
        X_test_2 = X_test[crop_mask_test]
        y_train_2 = np.where(y_train[crop_mask_train] == self.config.data.healthy_idx, 1, 0)
        y_test_2 = np.where(y_test[crop_mask_test] == self.config.data.healthy_idx, 1, 0)
        
        # Split 3: Disease classification (diseases only)
        disease_mask_train = np.isin(y_train, self.config.data.disease_indices)
        disease_mask_test = np.isin(y_test, self.config.data.disease_indices)
        
        X_train_3 = X_train[disease_mask_train]
        X_test_3 = X_test[disease_mask_test]
        y_train_3 = y_train[disease_mask_train]
        y_test_3 = y_test[disease_mask_test]
        
        # Map disease labels to sequential indices
        disease_label_map = {disease_idx: i for i, disease_idx in enumerate(self.config.data.disease_indices)}
        y_train_3_mapped = np.array([disease_label_map[label] for label in y_train_3])
        y_test_3_mapped = np.array([disease_label_map[label] for label in y_test_3])
        
        splits = {
            'classifier1': (X_train, X_test, y_train_1, y_test_1),
            'classifier2': (X_train_2, X_test_2, y_train_2, y_test_2),
            'classifier3': (X_train_3, X_test_3, y_train_3_mapped, y_test_3_mapped),
            'full_test': (X_test, y_test),
            'disease_label_map': disease_label_map
        }
        
        # Log split sizes
        self.logger.info(f"📊 Hierarchical split sizes:")
        self.logger.info(f"  - Classifier 1 (Crop vs Weed): {len(X_train)} train, {len(X_test)} test")
        self.logger.info(f"  - Classifier 2 (Healthy vs Disease): {len(X_train_2)} train, {len(X_test_2)} test")
        self.logger.info(f"  - Classifier 3 (Disease Types): {len(X_train_3)} train, {len(X_test_3)} test")
        
        return splits


class DataPreprocessingPipeline:
    """Main preprocessing pipeline that orchestrates all preprocessing steps."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.hierarchical_builder = HierarchicalDatasetBuilder(config)
        self.classical_preprocessor = ClassicalMLPreprocessor(config)
        self.dataset_splitter = DatasetSplitter(config)
    
    def run_deep_learning_preprocessing(self) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline for deep learning.
        
        Returns:
            Dictionary containing datasets and data loaders
        """
        self.logger.info("🚀 Starting deep learning preprocessing pipeline...")
        
        # Build hierarchical datasets
        datasets = self.hierarchical_builder.build_datasets()
        
        # Create data loaders
        data_loaders = self.hierarchical_builder.create_data_loaders(datasets)
        
        # Combine results
        result = {
            'datasets': datasets,
            'data_loaders': data_loaders
        }
        
        # Save preprocessing info
        self._save_preprocessing_info(result, 'deep_learning')
        
        self.logger.info("✅ Deep learning preprocessing completed!")
        return result
    
    def run_classical_ml_preprocessing(self) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline for classical ML.
        
        Returns:
            Dictionary containing file paths and hierarchical splits
        """
        self.logger.info("🚀 Starting classical ML preprocessing pipeline...")
        
        # Prepare file paths and labels
        file_paths, labels = self.classical_preprocessor.prepare_classical_ml_data()
        
        # Create train-test split
        X_train, X_test, y_train, y_test = self.dataset_splitter.create_stratified_split(
            file_paths, labels
        )
        
        # Create hierarchical splits
        hierarchical_splits = self.dataset_splitter.create_hierarchical_splits(
            X_train, X_test, y_train, y_test
        )
        
        result = {
            'file_paths': file_paths,
            'labels': labels,
            'train_test_split': (X_train, X_test, y_train, y_test),
            'hierarchical_splits': hierarchical_splits,
            'preprocessor': self.classical_preprocessor
        }
        
        # Save preprocessing info
        self._save_preprocessing_info(result, 'classical_ml')
        
        self.logger.info("✅ Classical ML preprocessing completed!")
        return result
    
    def _save_preprocessing_info(self, result: Dict[str, Any], pipeline_type: str):
        """Save preprocessing information to file."""
        info = {
            'pipeline_type': pipeline_type,
            'config': {
                'img_size': self.config.data.img_size,
                'test_size': self.config.data.test_size,
                'class_names': self.config.data.class_names,
                'healthy_idx': self.config.data.healthy_idx,
                'weed_idx': self.config.data.weed_idx,
                'disease_indices': self.config.data.disease_indices
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if pipeline_type == 'deep_learning' and 'datasets' in result:
            info['dataset_sizes'] = {
                'train_ds1': len(result['datasets']['train_ds1']),
                'train_ds2': len(result['datasets']['train_ds2']),
                'train_ds3': len(result['datasets']['train_ds3']),
                'test_ds1': len(result['datasets']['test_ds1']),
                'test_ds2': len(result['datasets']['test_ds2']),
                'test_ds3': len(result['datasets']['test_ds3'])
            }
        
        elif pipeline_type == 'classical_ml' and 'hierarchical_splits' in result:
            splits = result['hierarchical_splits']
            info['split_sizes'] = {
                'classifier1_train': len(splits['classifier1'][2]),
                'classifier1_test': len(splits['classifier1'][3]),
                'classifier2_train': len(splits['classifier2'][2]),
                'classifier2_test': len(splits['classifier2'][3]),
                'classifier3_train': len(splits['classifier3'][2]),
                'classifier3_test': len(splits['classifier3'][3])
            }
        
        # Save to file
        info_path = os.path.join(
            self.config.data.processed_data_dir, 
            f'{pipeline_type}_preprocessing_info.json'
        )
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        self.logger.info(f"📁 Preprocessing info saved to: {info_path}")


# Utility functions for standalone usage
def create_basic_transforms(img_size: int = 224) -> transforms.Compose:
    """Create basic image transforms."""
    return DataAugmentationFactory.create_basic_transform(img_size)


def create_augmented_transforms(img_size: int = 224) -> transforms.Compose:
    """Create augmented image transforms."""
    return DataAugmentationFactory.create_augmented_transform(img_size)


def preprocess_single_image(image_path: str, 
                          transform: transforms.Compose = None,
                          img_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the image
        transform: Transform to apply (if None, creates basic transform)
        img_size: Target image size
        
    Returns:
        Preprocessed image tensor
    """
    if transform is None:
        transform = create_basic_transforms(img_size)
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension
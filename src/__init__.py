# # src/__init__.py
# """
# Lettuce Disease Classification System
# A comprehensive AI system for agricultural disease detection using hierarchical classification.
# """

# __version__ = "1.0.0"
# __author__ = "Your Name"
# __email__ = "your.email@example.com"

# # src/utils/__init__.py
# """Utility modules for configuration, logging, and helpers."""

# from .config import Config, get_config, default_config
# from .logger import get_logger, setup_logging, create_session_logger

# __all__ = ['Config', 'get_config', 'default_config', 'get_logger', 'setup_logging', 'create_session_logger']

# # src/data_processing/__init__.py
# """Data processing modules for exploration and preprocessing."""

# from .data_exploration import DatasetExplorer
# from .preprocessing import (
#     DataPreprocessingPipeline, 
#     HierarchicalDatasetBuilder,
#     DataAugmentationFactory,
#     preprocess_single_image
# )

# __all__ = [
#     'DatasetExplorer',
#     'DataPreprocessingPipeline',
#     'HierarchicalDatasetBuilder', 
#     'DataAugmentationFactory',
#     'preprocess_single_image'
# ]

# # src/feature_extraction/__init__.py
# """Feature extraction modules for classical machine learning."""

# from .feature_extractor import (
#     ComprehensiveFeatureExtractor,
#     ColorFeatureExtractor,
#     TextureFeatureExtractor,
#     ShapeFeatureExtractor,
#     HOGFeatureExtractor,
#     extract_features_batch
# )

# __all__ = [
#     'ComprehensiveFeatureExtractor',
#     'ColorFeatureExtractor',
#     'TextureFeatureExtractor', 
#     'ShapeFeatureExtractor',
#     'HOGFeatureExtractor',
#     'extract_features_batch'
# ]

# # src/classical_ml/__init__.py
# """Classical machine learning modules."""

# from .hierarchical_classifier import HierarchicalClassicalML, HierarchicalClassifier

# __all__ = ['HierarchicalClassicalML', 'HierarchicalClassifier']

# # src/deep_learning/__init__.py
# """Deep learning modules."""

# # Note: These will be implemented based on your existing code
# # For now, we'll create placeholder imports

# __all__ = []

# # src/evaluation/__init__.py
# """Evaluation and metrics modules."""

# # Note: These will be implemented based on requirements
# # For now, we'll create placeholder imports

# __all__ = []
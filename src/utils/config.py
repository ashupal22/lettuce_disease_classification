"""
Configuration module for lettuce disease classification project.
Centralizes all configuration settings following SOLID principles.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    
    # Dataset paths
    raw_data_dir: str = "data/raw/Lettuce_disease_datasets"
    processed_data_dir: str = "data/processed"
    
    # Image processing
    img_size: int = 224
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Class information
    class_names: List[str] = None
    healthy_class: str = "Healthy"
    weed_class: str = "Shepherd_purse_weeds"
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                "Bacterial",
                "Downy_mildew_on_lettuce", 
                "Healthy",
                "Powdery_mildew_on_lettuce",
                "Septoria_blight_on_lettuce",
                "Shepherd_purse_weeds",
                "Viral",
                "Wilt_and_leaf_blight_on_lettuce"
            ]
    
    @property
    def healthy_idx(self) -> int:
        return self.class_names.index(self.healthy_class)
    
    @property
    def weed_idx(self) -> int:
        return self.class_names.index(self.weed_class)
    
    @property
    def disease_indices(self) -> List[int]:
        return [i for i, name in enumerate(self.class_names) 
                if name not in [self.healthy_class, self.weed_class]]
    
    @property
    def disease_names(self) -> List[str]:
        return [self.class_names[i] for i in self.disease_indices]


@dataclass
class ClassicalMLConfig:
    """Classical ML configuration."""
    
    # Feature extraction
    lbp_radius: int = 3
    lbp_n_points: int = 24
    glcm_distances: List[int] = None
    glcm_angles: List[float] = None
    color_hist_bins: int = 32
    
    # Model parameters
    random_forest_n_estimators: int = 100
    svm_kernel: str = "rbf"
    cv_folds: int = 3
    
    # Model paths
    models_dir: str = "models/classical_ml"
    
    def __post_init__(self):
        if self.glcm_distances is None:
            self.glcm_distances = [5]
        if self.glcm_angles is None:
            import numpy as np
            self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]


@dataclass
class DeepLearningConfig:
    """Deep learning configuration."""
    
    # Model architecture
    num_classes_binary: int = 1
    num_classes_disease: int = 6
    dropout_rate: float = 0.5
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Data augmentation
    rotation_degrees: int = 20
    horizontal_flip_prob: float = 0.5
    crop_scale_min: float = 0.8
    crop_scale_max: float = 1.0
    
    # Loss function parameters
    focal_loss_gamma_pos: float = 1.0
    focal_loss_gamma_neg: float = 4.0
    
    # Model paths
    models_dir: str = "models/deep_learning"
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Metrics
    metrics: List[str] = None
    confusion_matrix_figsize: Tuple[int, int] = (12, 10)
    classification_report_digits: int = 4
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_training_curves: bool = True
    plot_feature_importance: bool = True
    save_plots: bool = True
    
    # Output paths
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    reports_dir: str = "results/reports"
    metrics_dir: str = "results/metrics"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix"]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "results/logs/training.log"
    console_output: bool = True


class Config:
    """Main configuration class that combines all configurations."""
    
    def __init__(self, 
                 data_config: DataConfig = None,
                 classical_ml_config: ClassicalMLConfig = None,
                 deep_learning_config: DeepLearningConfig = None,
                 evaluation_config: EvaluationConfig = None,
                 logging_config: LoggingConfig = None):
        
        self.data = data_config or DataConfig()
        self.classical_ml = classical_ml_config or ClassicalMLConfig()
        self.deep_learning = deep_learning_config or DeepLearningConfig()
        self.evaluation = evaluation_config or EvaluationConfig()
        self.logging = logging_config or LoggingConfig()
        
        # Global settings
        self.seed = 42
        self.project_root = Path(__file__).parent.parent.parent
        
        # Create necessary directories
        self._create_directories()
        
        # Set device for deep learning
        self._set_device()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.processed_data_dir,
            self.classical_ml.models_dir,
            self.deep_learning.models_dir,
            self.evaluation.results_dir,
            self.evaluation.plots_dir,
            self.evaluation.reports_dir,
            self.evaluation.metrics_dir,
            os.path.dirname(self.logging.log_file)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _set_device(self):
        """Set the appropriate device for deep learning."""
        if self.deep_learning.device == "auto":
            import torch
            self.deep_learning.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data-related paths."""
        return {
            "raw_data": self.data.raw_data_dir,
            "processed_data": self.data.processed_data_dir,
            "classical_models": self.classical_ml.models_dir,
            "deep_models": self.deep_learning.models_dir,
            "results": self.evaluation.results_dir,
            "plots": self.evaluation.plots_dir,
            "reports": self.evaluation.reports_dir,
            "metrics": self.evaluation.metrics_dir
        }
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist or can be created."""
        try:
            if not os.path.exists(self.data.raw_data_dir):
                raise FileNotFoundError(f"Raw data directory not found: {self.data.raw_data_dir}")
            
            # Check if class directories exist
            for class_name in self.data.class_names:
                class_dir = os.path.join(self.data.raw_data_dir, class_name)
                if not os.path.exists(class_dir):
                    raise FileNotFoundError(f"Class directory not found: {class_dir}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"Path validation failed: {e}")
            return False
    
    def save_config(self, filepath: str = None):
        """Save configuration to a file."""
        import json
        
        if filepath is None:
            filepath = os.path.join(self.evaluation.results_dir, "config.json")
        
        config_dict = {
            "data": self.data.__dict__,
            "classical_ml": self.classical_ml.__dict__,
            "deep_learning": self.deep_learning.__dict__,
            "evaluation": self.evaluation.__dict__,
            "logging": self.logging.__dict__,
            "seed": self.seed
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        
        # Process the config dict
        for section in config_dict.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    section[key] = convert_numpy(value)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from a file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct numpy arrays
        import numpy as np
        if 'classical_ml' in config_dict and 'glcm_angles' in config_dict['classical_ml']:
            config_dict['classical_ml']['glcm_angles'] = np.array(config_dict['classical_ml']['glcm_angles'])
        
        # Create individual config objects
        data_config = DataConfig(**config_dict['data'])
        classical_ml_config = ClassicalMLConfig(**config_dict['classical_ml'])
        deep_learning_config = DeepLearningConfig(**config_dict['deep_learning'])
        evaluation_config = EvaluationConfig(**config_dict['evaluation'])
        logging_config = LoggingConfig(**config_dict['logging'])
        
        config = cls(
            data_config=data_config,
            classical_ml_config=classical_ml_config,
            deep_learning_config=deep_learning_config,
            evaluation_config=evaluation_config,
            logging_config=logging_config
        )
        
        config.seed = config_dict.get('seed', 42)
        
        return config
    
    def __str__(self) -> str:
        """String representation of configuration."""
        lines = [
            "🌿 Lettuce Disease Classification Configuration",
            "=" * 50,
            f"Project Root: {self.project_root}",
            f"Seed: {self.seed}",
            f"Device: {self.deep_learning.device}",
            "",
            "📊 Data Configuration:",
            f"  Raw Data: {self.data.raw_data_dir}",
            f"  Image Size: {self.data.img_size}x{self.data.img_size}",
            f"  Classes: {len(self.data.class_names)}",
            f"  Test Size: {self.data.test_size}",
            "",
            "🔬 Classical ML Configuration:",
            f"  Models Directory: {self.classical_ml.models_dir}",
            f"  Random Forest Estimators: {self.classical_ml.random_forest_n_estimators}",
            f"  SVM Kernel: {self.classical_ml.svm_kernel}",
            "",
            "🧠 Deep Learning Configuration:",
            f"  Models Directory: {self.deep_learning.models_dir}",
            f"  Batch Size: {self.deep_learning.batch_size}",
            f"  Epochs: {self.deep_learning.epochs}",
            f"  Learning Rate: {self.deep_learning.learning_rate}",
            "",
            "📈 Evaluation Configuration:",
            f"  Results Directory: {self.evaluation.results_dir}",
            f"  Save Plots: {self.evaluation.save_plots}",
            f"  Metrics: {', '.join(self.evaluation.metrics)}",
            "=" * 50
        ]
        return "\n".join(lines)


# Default configuration instance
default_config = Config()


# Utility function to get configuration
def get_config(config_path: str = None) -> Config:
    """Get configuration instance."""
    if config_path and os.path.exists(config_path):
        return Config.load_config(config_path)
    return default_config
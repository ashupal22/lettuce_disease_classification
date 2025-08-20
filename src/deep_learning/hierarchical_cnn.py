"""
Hierarchical Deep Learning CNN for lettuce disease classification.
A simplified wrapper around your existing deep learning code.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..utils.config import Config
from ..utils.logger import get_logger, ModelTrainingLogger


class SimpleCNN(nn.Module):
    """Simple CNN architecture for hierarchical classification."""
    
    def __init__(self, num_classes: int, img_size: int = 224, dropout_rate: float = 0.5):
        """
        Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes
            img_size: Input image size
            dropout_rate: Dropout probability
        """
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = 128 * (img_size // 8) * (img_size // 8)
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class WeightedAsymmetricLoss(nn.Module):
    """Weighted asymmetric loss for handling class imbalance."""
    
    def __init__(self, weight=None, gamma_pos=1, gamma_neg=4):
        super().__init__()
        self.weight = weight
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets):
        if logits.size(1) == 1:
            # Binary classification
            targets = targets.float().unsqueeze(1)
        else:
            # Multi-class classification
            targets = F.one_hot(targets, num_classes=logits.size(1)).float().to(logits.device)
        
        prob = torch.sigmoid(logits)
        pos_loss = -targets * ((1 - prob) ** self.gamma_pos) * torch.log(prob + 1e-8)
        neg_loss = -(1 - targets) * (prob ** self.gamma_neg) * torch.log(1 - prob + 1e-8)
        
        loss = pos_loss + neg_loss
        
        if self.weight is not None:
            loss = loss * self.weight
        
        return loss.mean()


class HierarchicalCNNTrainer:
    """Trainer for individual CNN classifiers."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 model_name: str):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            model_name: Name of the model
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.logger = get_logger(__name__)
        
        # Move model to device
        self.model.to(device)
    
    def train(self, epochs: int) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        training_logger = ModelTrainingLogger(self.logger, self.model_name)
        training_logger.start_training(epochs)
        
        self.model.train()
        training_history = {'losses': [], 'accuracies': []}
        
        for epoch in range(epochs):
            training_logger.start_epoch(epoch + 1, epochs)
            
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                # Calculate loss
                if outputs.size(1) == 1:
                    # Binary classification
                    loss = self.criterion(outputs, targets.unsqueeze(1).float())
                    predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                    correct += (predictions == targets).sum().item()
                else:
                    # Multi-class classification
                    loss = self.criterion(outputs, targets)
                    _, predictions = torch.max(outputs, 1)
                    correct += (predictions == targets).sum().item()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total += targets.size(0)
                
                # Log batch progress occasionally
                if batch_idx % max(1, len(self.train_loader) // 10) == 0:
                    training_logger.log_batch_progress(
                        batch_idx, len(self.train_loader), 
                        loss.item(), correct / total
                    )
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct / total
            
            training_history['losses'].append(avg_loss)
            training_history['accuracies'].append(accuracy)
            
            training_logger.end_epoch(epoch + 1, avg_loss, accuracy)
        
        final_metrics = {
            'final_loss': training_history['losses'][-1],
            'final_accuracy': training_history['accuracies'][-1]
        }
        
        training_logger.end_training(final_metrics)
        
        return training_history


class HierarchicalCNN:
    """Main hierarchical CNN system."""
    
    def __init__(self, config: Config):
        """
        Initialize hierarchical CNN system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.device = config.deep_learning.device
        
        # Models
        self.model1 = None  # Crop vs Weed
        self.model2 = None  # Healthy vs Disease
        self.model3 = None  # Disease Classification
        
        # Training history
        self.training_history = {}
        self.evaluation_results = {}
    
    def build_models(self):
        """Build the three CNN models."""
        self.logger.info("🏗️ Building CNN models...")
        
        # Model 1: Binary classification (Crop vs Weed)
        self.model1 = SimpleCNN(
            num_classes=1,
            img_size=self.config.data.img_size,
            dropout_rate=self.config.deep_learning.dropout_rate
        )
        
        # Model 2: Binary classification (Healthy vs Disease)
        self.model2 = SimpleCNN(
            num_classes=1,
            img_size=self.config.data.img_size,
            dropout_rate=self.config.deep_learning.dropout_rate
        )
        
        # Model 3: Multi-class classification (Disease types)
        self.model3 = SimpleCNN(
            num_classes=len(self.config.data.disease_indices),
            img_size=self.config.data.img_size,
            dropout_rate=self.config.deep_learning.dropout_rate
        )
        
        self.logger.info("✅ Models built successfully")
    
    def train_models(self, data_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Train all three models.
        
        Args:
            data_loaders: Dictionary containing training data loaders
            
        Returns:
            Training results for all models
        """
        self.logger.info("🚀 Training Hierarchical CNN System")
        self.logger.info("=" * 60)
        
        if self.model1 is None:
            self.build_models()
        
        training_results = {}
        
        # Train Model 1: Crop vs Weed
        self.logger.info("\n🌱 TRAINING MODEL 1: CROP vs WEED")
        model1_results = self._train_single_model(
            self.model1,
            data_loaders['train_loader1'],
            "Model 1 (Crop vs Weed)",
            binary=True
        )
        training_results['model1'] = model1_results
        
        # Train Model 2: Healthy vs Disease
        self.logger.info("\n🏥 TRAINING MODEL 2: HEALTHY vs DISEASE")
        model2_results = self._train_single_model(
            self.model2,
            data_loaders['train_loader2'],
            "Model 2 (Healthy vs Disease)",
            binary=True
        )
        training_results['model2'] = model2_results
        
        # Train Model 3: Disease Classification
        self.logger.info("\n🦠 TRAINING MODEL 3: DISEASE CLASSIFICATION")
        model3_results = self._train_single_model(
            self.model3,
            data_loaders['train_loader3'],
            "Model 3 (Disease Classification)",
            binary=False
        )
        training_results['model3'] = model3_results
        
        self.training_history = training_results
        
        self.logger.info("\n✅ HIERARCHICAL CNN TRAINING COMPLETED")
        self.logger.info("=" * 60)
        
        return training_results
    
    def _train_single_model(self, 
                           model: nn.Module,
                           train_loader: DataLoader,
                           model_name: str,
                           binary: bool = True) -> Dict[str, Any]:
        """Train a single model."""
        
        # Setup loss function
        if binary:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = WeightedAsymmetricLoss()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.deep_learning.learning_rate,
            weight_decay=self.config.deep_learning.weight_decay
        )
        
        # Create trainer
        trainer = HierarchicalCNNTrainer(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=self.device,
            model_name=model_name
        )
        
        # Train model
        training_history = trainer.train(self.config.deep_learning.epochs)
        
        return training_history
    
    def evaluate_models(self, data_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            data_loaders: Dictionary containing test data loaders
            
        Returns:
            Evaluation results for all models
        """
        self.logger.info("📊 Evaluating Hierarchical CNN System")
        self.logger.info("=" * 60)
        
        evaluation_results = {}
        
        # Evaluate individual models
        if self.model1 is not None:
            eval1 = self._evaluate_single_model(
                self.model1, data_loaders['test_loader1'], 
                "Model 1 (Crop vs Weed)", binary=True
            )
            evaluation_results['model1'] = eval1
        
        if self.model2 is not None:
            eval2 = self._evaluate_single_model(
                self.model2, data_loaders['test_loader2'],
                "Model 2 (Healthy vs Disease)", binary=True
            )
            evaluation_results['model2'] = eval2
        
        if self.model3 is not None:
            eval3 = self._evaluate_single_model(
                self.model3, data_loaders['test_loader3'],
                "Model 3 (Disease Classification)", binary=False
            )
            evaluation_results['model3'] = eval3
        
        # Evaluate hierarchical system
        if 'test_loader_full' in data_loaders:
            hierarchical_eval = self._evaluate_hierarchical_system(
                data_loaders['test_loader_full']
            )
            evaluation_results['hierarchical'] = hierarchical_eval
        
        self.evaluation_results = evaluation_results
        
        self.logger.info("✅ CNN Evaluation completed")
        
        return evaluation_results
    
    def _evaluate_single_model(self, 
                              model: nn.Module,
                              test_loader: DataLoader,
                              model_name: str,
                              binary: bool = True) -> Dict[str, Any]:
        """Evaluate a single model."""
        
        self.logger.info(f"📊 Evaluating {model_name}")
        
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                
                if binary:
                    predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                else:
                    _, predictions = torch.max(outputs, 1)
                
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Create class labels
        if binary:
            class_labels = ['Negative', 'Positive']
        else:
            class_labels = [f'Class_{i}' for i in range(len(self.config.data.disease_indices))]
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred,
            target_names=class_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info(f"  📈 {model_name} Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'classification_report': class_report,
            'confusion_matrix': cm
        }
    
    def _evaluate_hierarchical_system(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the complete hierarchical system."""
        
        self.logger.info("🔄 Evaluating hierarchical system...")
        
        if not all([self.model1, self.model2, self.model3]):
            self.logger.warning("Not all models are available for hierarchical evaluation")
            return {}
        
        # Set models to evaluation mode
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Stage 1: Crop vs Weed
                out1 = torch.sigmoid(self.model1(data)).cpu().numpy().ravel()
                is_crop = out1 > 0.5
                
                batch_predictions = np.zeros(batch_size, dtype=int)
                
                for i in range(batch_size):
                    if not is_crop[i]:
                        # Predicted as weed
                        batch_predictions[i] = self.config.data.weed_idx
                    else:
                        # Predicted as crop, go to stage 2
                        sample = data[i:i+1]
                        
                        # Stage 2: Healthy vs Disease
                        out2 = torch.sigmoid(self.model2(sample)).cpu().item()
                        
                        if out2 > 0.5:
                            # Predicted as healthy
                            batch_predictions[i] = self.config.data.healthy_idx
                        else:
                            # Predicted as disease, go to stage 3
                            out3 = self.model3(sample)
                            disease_pred = torch.argmax(out3, dim=1).cpu().item()
                            
                            # Map back to original disease index
                            batch_predictions[i] = self.config.data.disease_indices[disease_pred]
                
                y_true.extend(targets.numpy())
                y_pred.extend(batch_predictions)
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred,
            target_names=self.config.data.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.config.data.class_names))))
        
        self.logger.info(f"🎯 Overall Hierarchical CNN Accuracy: {overall_accuracy:.4f}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'classification_report': class_report,
            'confusion_matrix': cm
        }
    
    def save_models(self, base_dir: str = None):
        """Save all trained models."""
        if base_dir is None:
            base_dir = self.config.deep_learning.models_dir
        
        os.makedirs(base_dir, exist_ok=True)
        
        if self.model1 is not None:
            torch.save(self.model1.state_dict(), os.path.join(base_dir, 'model1.pth'))
        
        if self.model2 is not None:
            torch.save(self.model2.state_dict(), os.path.join(base_dir, 'model2.pth'))
        
        if self.model3 is not None:
            torch.save(self.model3.state_dict(), os.path.join(base_dir, 'model3.pth'))
        
        # Save metadata
        metadata = {
            'config': {
                'img_size': self.config.data.img_size,
                'dropout_rate': self.config.deep_learning.dropout_rate,
                'class_names': self.config.data.class_names,
                'disease_indices': self.config.data.disease_indices
            },
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results
        }
        
        torch.save(metadata, os.path.join(base_dir, 'metadata.pth'))
        
        self.logger.info(f"💾 All CNN models saved to: {base_dir}")
    
    def load_models(self, base_dir: str = None):
        """Load all trained models."""
        if base_dir is None:
            base_dir = self.config.deep_learning.models_dir
        
        # Build models first
        self.build_models()
        
        # Load model states
        model1_path = os.path.join(base_dir, 'model1.pth')
        if os.path.exists(model1_path):
            self.model1.load_state_dict(torch.load(model1_path, map_location=self.device))
        
        model2_path = os.path.join(base_dir, 'model2.pth')
        if os.path.exists(model2_path):
            self.model2.load_state_dict(torch.load(model2_path, map_location=self.device))
        
        model3_path = os.path.join(base_dir, 'model3.pth')
        if os.path.exists(model3_path):
            self.model3.load_state_dict(torch.load(model3_path, map_location=self.device))
        
        # Load metadata
        metadata_path = os.path.join(base_dir, 'metadata.pth')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location=self.device)
            self.training_history = metadata.get('training_history', {})
            self.evaluation_results = metadata.get('evaluation_results', {})
        
        self.logger.info(f"📂 CNN models loaded from: {base_dir}")
    
    def run_full_pipeline(self, data_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Run the complete deep learning pipeline.
        
        Args:
            data_loaders: Dictionary containing all data loaders
            
        Returns:
            Complete results including training and evaluation
        """
        self.logger.info("🚀 Running Complete Deep Learning Pipeline")
        
        # Training
        training_results = self.train_models(data_loaders)
        
        # Evaluation
        evaluation_results = self.evaluate_models(data_loaders)
        
        # Save models
        self.save_models()
        
        # Create visualizations
        self._create_visualizations()
        
        # Combine results
        complete_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'final_accuracy': evaluation_results.get('hierarchical', {}).get('overall_accuracy', 0.0)
        }
        
        self.logger.info(f"🎉 Deep Learning Pipeline Completed!")
        if 'final_accuracy' in complete_results:
            self.logger.info(f"🎯 Final Accuracy: {complete_results['final_accuracy']:.4f}")
        
        return complete_results
    
    def _create_visualizations(self):
        """Create visualizations for deep learning results."""
        if not self.evaluation_results:
            return
        
        self.logger.info("📊 Creating CNN visualizations...")
        
        # Plot training curves
        self._plot_training_curves()
        
        # Plot confusion matrix for hierarchical system
        if 'hierarchical' in self.evaluation_results:
            self._plot_confusion_matrix()
        
        self.logger.info("✅ CNN Visualizations created")
    
    def _plot_training_curves(self):
        """Plot training loss and accuracy curves."""
        if not self.training_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (model_name, history) in enumerate(self.training_history.items()):
            if 'losses' in history and 'accuracies' in history:
                col = idx
                
                # Plot loss
                axes[0, col].plot(history['losses'])
                axes[0, col].set_title(f'{model_name.title()} - Training Loss')
                axes[0, col].set_xlabel('Epoch')
                axes[0, col].set_ylabel('Loss')
                axes[0, col].grid(True)
                
                # Plot accuracy
                axes[1, col].plot(history['accuracies'])
                axes[1, col].set_title(f'{model_name.title()} - Training Accuracy')
                axes[1, col].set_xlabel('Epoch')
                axes[1, col].set_ylabel('Accuracy')
                axes[1, col].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'cnn_training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Plot saved: {plot_path}")
        
        plt.show()
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix for hierarchical system."""
        if 'hierarchical' not in self.evaluation_results:
            return
        
        hierarchical_results = self.evaluation_results['hierarchical']
        if 'confusion_matrix' not in hierarchical_results:
            return
        
        cm = hierarchical_results['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.data.class_names,
                   yticklabels=self.config.data.class_names)
        
        plt.title('Deep Learning CNN - Hierarchical System Confusion Matrix', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'cnn_confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Plot saved: {plot_path}")
        
        plt.show()
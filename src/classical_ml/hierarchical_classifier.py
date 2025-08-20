"""
Hierarchical Classical ML Classifier for lettuce disease classification.
Implements the three-stage hierarchical approach using traditional ML algorithms.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, GridSearchCV
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

from ..utils.config import Config
from ..utils.logger import get_logger, ModelTrainingLogger, ProgressLogger
from ..feature_extraction.feature_extractor import ComprehensiveFeatureExtractor


class HierarchicalClassifier:
    """Individual classifier in the hierarchical system."""
    
    def __init__(self, 
                 classifier_name: str,
                 class_labels: List[str],
                 config: Config):
        """
        Initialize hierarchical classifier.
        
        Args:
            classifier_name: Name of this classifier
            class_labels: Labels for the classes this classifier handles
            config: Configuration object
        """
        self.classifier_name = classifier_name
        self.class_labels = class_labels
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model and preprocessing
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Performance tracking
        self.training_history = {}
        self.best_params = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the classifier with cross-validation and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training results and metrics
        """
        training_logger = ModelTrainingLogger(self.logger, self.classifier_name)
        training_logger.start_training(1)  # Classical ML doesn't have epochs
        
        self.logger.info(f"🎯 Training {self.classifier_name}")
        self.logger.info(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define models to try
        models = self._get_candidate_models()
        
        # Cross-validation to select best model
        best_model, best_score, cv_results = self._select_best_model(
            models, X_train_scaled, y_train
        )
        
        # Train final model
        self.model = best_model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        training_metrics = self._calculate_metrics(y_train, y_pred_train)
        
        # Store training history
        self.training_history = {
            'cv_results': cv_results,
            'best_score': best_score,
            'training_metrics': training_metrics,
            'model_type': type(best_model).__name__
        }
        
        training_logger.end_training(training_metrics)
        
        self.logger.info(f"✅ {self.classifier_name} training completed")
        self.logger.info(f"🏆 Best model: {type(best_model).__name__}")
        self.logger.info(f"📈 CV Score: {best_score:.4f}")
        self.logger.info(f"📊 Training Accuracy: {training_metrics['accuracy']:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError(f"{self.classifier_name} is not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError(f"{self.classifier_name} is not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError(f"{self.classifier_name} is not fitted yet")
        
        self.logger.info(f"📊 Evaluating {self.classifier_name}")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Detailed classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        # Log results
        self.logger.info(f"📈 {self.classifier_name} Test Results:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        if len(self.class_labels) == 2:
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        else:
            self.logger.info(f"  Macro F1: {metrics['f1_macro']:.4f}")
            self.logger.info(f"  Weighted F1: {metrics['f1_weighted']:.4f}")
        
        return results
    
    def _get_candidate_models(self) -> Dict[str, Any]:
        """Get candidate models for selection."""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=self.config.classical_ml.random_forest_n_estimators,
                random_state=self.config.seed,
                class_weight='balanced',
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel=self.config.classical_ml.svm_kernel,
                random_state=self.config.seed,
                class_weight='balanced',
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                random_state=self.config.seed,
                class_weight='balanced',
                max_iter=1000,
                n_jobs=-1
            )
        }
        
        return models
    
    def _select_best_model(self, 
                          models: Dict[str, Any], 
                          X_train: np.ndarray, 
                          y_train: np.ndarray) -> Tuple[Any, float, Dict[str, float]]:
        """Select best model using cross-validation."""
        self.logger.info("🔍 Performing cross-validation for model selection...")
        
        cv_results = {}
        best_score = 0
        best_model = None
        
        # Determine scoring metric
        scoring = 'f1_weighted' if len(np.unique(y_train)) > 2 else 'f1'
        
        for name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=self.config.classical_ml.cv_folds,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                mean_score = scores.mean()
                cv_results[name] = {
                    'mean_score': mean_score,
                    'std_score': scores.std(),
                    'all_scores': scores.tolist()
                }
                
                self.logger.info(f"  {name:15} | CV {scoring}: {mean_score:.4f} ± {scores.std():.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    
            except Exception as e:
                self.logger.warning(f"  {name:15} | Error: {str(e)}")
                cv_results[name] = {'error': str(e)}
        
        if best_model is None:
            # Fallback to Random Forest
            best_model = models['RandomForest']
            self.logger.warning("No model succeeded in CV, using Random Forest as fallback")
        
        return best_model, best_score, cv_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        if len(np.unique(y_true)) == 2:
            # Binary classification
            metrics.update({
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            })
        else:
            # Multi-class classification
            metrics.update({
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            })
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classifier_name': self.classifier_name,
            'class_labels': self.class_labels,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.classifier_name = model_data['classifier_name']
        self.class_labels = model_data['class_labels']
        self.training_history = model_data.get('training_history', {})
        self.is_fitted = model_data.get('is_fitted', True)
        
        self.logger.info(f"📂 Model loaded from: {filepath}")


class HierarchicalClassicalML:
    """Main hierarchical classical ML system."""
    
    def __init__(self, config: Config):
        """
        Initialize hierarchical classical ML system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Feature extractor
        self.feature_extractor = ComprehensiveFeatureExtractor(config)
        
        # Three-stage classifiers
        self.classifier1 = HierarchicalClassifier(
            "Classifier 1: Crop vs Weed",
            ["Weed", "Crop"],
            config
        )
        
        self.classifier2 = HierarchicalClassifier(
            "Classifier 2: Healthy vs Disease", 
            ["Disease", "Healthy"],
            config
        )
        
        disease_names = [config.data.class_names[i] for i in config.data.disease_indices]
        self.classifier3 = HierarchicalClassifier(
            "Classifier 3: Disease Classification",
            disease_names,
            config
        )
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
    
    def extract_features_from_paths(self, 
                                   file_paths: np.ndarray, 
                                   labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from image file paths.
        
        Args:
            file_paths: Array of image file paths
            labels: Corresponding labels
            
        Returns:
            Tuple of (features, labels) for successful extractions
        """
        self.logger.info("🔍 Extracting features from images...")
        
        features_list = []
        labels_list = []
        failed_count = 0
        
        progress_logger = ProgressLogger(
            self.logger, len(file_paths), "Feature Extraction", log_interval=50
        )
        
        for i, (img_path, label) in enumerate(zip(file_paths, labels)):
            features = self.feature_extractor.extract_features(img_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
            else:
                failed_count += 1
            
            progress_logger.update()
        
        progress_logger.finish()
        
        if failed_count > 0:
            self.logger.warning(f"⚠️ Failed to extract features from {failed_count} images")
        
        if not features_list:
            raise ValueError("No features could be extracted from any images")
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        self.logger.info(f"✅ Successfully extracted features from {len(features_array)} images")
        self.logger.info(f"📊 Feature matrix shape: {features_array.shape}")
        
        return features_array, labels_array
    
    def train_hierarchical_system(self, hierarchical_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the complete hierarchical system.
        
        Args:
            hierarchical_splits: Dictionary containing hierarchical data splits
            
        Returns:
            Training results for all classifiers
        """
        self.logger.info("🚀 Training Hierarchical Classical ML System")
        self.logger.info("=" * 60)
        
        # Extract features for all data first
        X_train, X_test, y_train, y_test = hierarchical_splits['full_test'][:4]
        
        # Extract features
        features_train, labels_train = self.extract_features_from_paths(X_train, y_train)
        features_test, labels_test = self.extract_features_from_paths(X_test, y_test)
        
        # Create hierarchical splits with features
        feature_splits = self._create_feature_splits(
            features_train, labels_train, features_test, labels_test, hierarchical_splits
        )
        
        # Train each classifier
        training_results = {}
        
        # Classifier 1: Crop vs Weed
        self.logger.info("\n" + "🌱 TRAINING CLASSIFIER 1: CROP vs WEED" + "\n" + "-" * 50)
        X_train_1, X_test_1, y_train_1, y_test_1 = feature_splits['classifier1']
        training_results['classifier1'] = self.classifier1.train(X_train_1, y_train_1)
        
        # Classifier 2: Healthy vs Disease
        self.logger.info("\n" + "🏥 TRAINING CLASSIFIER 2: HEALTHY vs DISEASE" + "\n" + "-" * 50)
        X_train_2, X_test_2, y_train_2, y_test_2 = feature_splits['classifier2']
        if len(np.unique(y_train_2)) > 1:
            training_results['classifier2'] = self.classifier2.train(X_train_2, y_train_2)
        else:
            self.logger.warning("Insufficient classes for Classifier 2")
            training_results['classifier2'] = {'error': 'Insufficient classes'}
        
        # Classifier 3: Disease Classification
        self.logger.info("\n" + "🦠 TRAINING CLASSIFIER 3: DISEASE CLASSIFICATION" + "\n" + "-" * 50)
        X_train_3, X_test_3, y_train_3, y_test_3 = feature_splits['classifier3']
        if len(np.unique(y_train_3)) > 1:
            training_results['classifier3'] = self.classifier3.train(X_train_3, y_train_3)
        else:
            self.logger.warning("Insufficient classes for Classifier 3")
            training_results['classifier3'] = {'error': 'Insufficient classes'}
        
        # Store training results
        self.training_results = training_results
        
        # Store feature splits for evaluation
        self.feature_splits = feature_splits
        
        self.logger.info("\n" + "✅ HIERARCHICAL TRAINING COMPLETED" + "\n" + "=" * 60)
        
        return training_results
    
    def _create_feature_splits(self, 
                              features_train: np.ndarray, 
                              labels_train: np.ndarray,
                              features_test: np.ndarray, 
                              labels_test: np.ndarray,
                              hierarchical_splits: Dict[str, Any]) -> Dict[str, Tuple]:
        """Create feature-based hierarchical splits."""
        
        # Split 1: Crop vs Weed
        y_train_1 = np.where(labels_train == self.config.data.weed_idx, 0, 1)
        y_test_1 = np.where(labels_test == self.config.data.weed_idx, 0, 1)
        
        # Split 2: Healthy vs Disease (crops only)
        crop_mask_train = labels_train != self.config.data.weed_idx
        crop_mask_test = labels_test != self.config.data.weed_idx
        
        features_train_2 = features_train[crop_mask_train]
        features_test_2 = features_test[crop_mask_test]
        y_train_2 = np.where(labels_train[crop_mask_train] == self.config.data.healthy_idx, 1, 0)
        y_test_2 = np.where(labels_test[crop_mask_test] == self.config.data.healthy_idx, 1, 0)
        
        # Split 3: Disease classification (diseases only)
        disease_mask_train = np.isin(labels_train, self.config.data.disease_indices)
        disease_mask_test = np.isin(labels_test, self.config.data.disease_indices)
        
        features_train_3 = features_train[disease_mask_train]
        features_test_3 = features_test[disease_mask_test]
        y_train_3 = labels_train[disease_mask_train]
        y_test_3 = labels_test[disease_mask_test]
        
        # Map disease labels to sequential indices
        disease_label_map = {disease_idx: i for i, disease_idx in enumerate(self.config.data.disease_indices)}
        y_train_3_mapped = np.array([disease_label_map[label] for label in y_train_3])
        y_test_3_mapped = np.array([disease_label_map[label] for label in y_test_3])
        
        return {
            'classifier1': (features_train, features_test, y_train_1, y_test_1),
            'classifier2': (features_train_2, features_test_2, y_train_2, y_test_2),
            'classifier3': (features_train_3, features_test_3, y_train_3_mapped, y_test_3_mapped),
            'full_test': (features_test, labels_test),
            'disease_label_map': disease_label_map
        }
    
    def evaluate_hierarchical_system(self) -> Dict[str, Any]:
        """
        Evaluate the complete hierarchical system.
        
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("📊 Evaluating Hierarchical Classical ML System")
        self.logger.info("=" * 60)
        
        evaluation_results = {}
        
        # Individual classifier evaluations
        if hasattr(self, 'feature_splits'):
            # Evaluate Classifier 1
            X_test_1, y_test_1 = self.feature_splits['classifier1'][1], self.feature_splits['classifier1'][3]
            if self.classifier1.is_fitted:
                evaluation_results['classifier1'] = self.classifier1.evaluate(X_test_1, y_test_1)
            
            # Evaluate Classifier 2
            X_test_2, y_test_2 = self.feature_splits['classifier2'][1], self.feature_splits['classifier2'][3]
            if self.classifier2.is_fitted and len(X_test_2) > 0:
                evaluation_results['classifier2'] = self.classifier2.evaluate(X_test_2, y_test_2)
            
            # Evaluate Classifier 3
            X_test_3, y_test_3 = self.feature_splits['classifier3'][1], self.feature_splits['classifier3'][3]
            if self.classifier3.is_fitted and len(X_test_3) > 0:
                evaluation_results['classifier3'] = self.classifier3.evaluate(X_test_3, y_test_3)
        
        # Hierarchical system evaluation
        if hasattr(self, 'feature_splits'):
            features_test, labels_test = self.feature_splits['full_test']
            hierarchical_results = self._evaluate_hierarchical_predictions(features_test, labels_test)
            evaluation_results['hierarchical'] = hierarchical_results
        
        self.evaluation_results = evaluation_results
        
        self.logger.info("✅ Evaluation completed")
        
        return evaluation_results
    
    def _evaluate_hierarchical_predictions(self, 
                                         features_test: np.ndarray, 
                                         labels_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the complete hierarchical system predictions."""
        self.logger.info("🔄 Generating hierarchical predictions...")
        
        predictions = []
        prediction_paths = []
        
        # Create reverse mapping for disease labels
        disease_label_map = {i: disease_idx for i, disease_idx in enumerate(self.config.data.disease_indices)}
        
        for i in range(len(features_test)):
            sample = features_test[i:i+1]
            path = []
            
            # Stage 1: Crop vs Weed
            if self.classifier1.is_fitted:
                crop_prob = self.classifier1.predict_proba(sample)[0]
                is_crop = self.classifier1.predict(sample)[0]
                
                if is_crop == 0:  # Weed
                    predictions.append(self.config.data.weed_idx)
                    path.append(f"Step1: Weed (prob={crop_prob[0]:.3f})")
                else:  # Crop
                    path.append(f"Step1: Crop (prob={crop_prob[1]:.3f})")
                    
                    # Stage 2: Healthy vs Disease
                    if self.classifier2.is_fitted:
                        healthy_prob = self.classifier2.predict_proba(sample)[0]
                        is_healthy = self.classifier2.predict(sample)[0]
                        
                        if is_healthy == 1:  # Healthy
                            predictions.append(self.config.data.healthy_idx)
                            path.append(f"Step2: Healthy (prob={healthy_prob[1]:.3f})")
                        else:  # Disease
                            path.append(f"Step2: Disease (prob={healthy_prob[0]:.3f})")
                            
                            # Stage 3: Disease type
                            if self.classifier3.is_fitted:
                                disease_probs = self.classifier3.predict_proba(sample)[0]
                                disease_type = self.classifier3.predict(sample)[0]
                                max_prob = np.max(disease_probs)
                                
                                # Map back to original disease index
                                original_disease_idx = disease_label_map[disease_type]
                                predictions.append(original_disease_idx)
                                
                                disease_name = self.config.data.class_names[original_disease_idx]
                                path.append(f"Step3: {disease_name} (prob={max_prob:.3f})")
                            else:
                                # Fallback
                                predictions.append(self.config.data.disease_indices[0])
                                path.append("Step3: Fallback to first disease")
                    else:
                        # Fallback
                        predictions.append(self.config.data.healthy_idx)
                        path.append("Step2: Fallback to healthy")
            else:
                # Fallback if no classifier is fitted
                predictions.append(0)
                path.append("Fallback: No fitted classifier")
            
            prediction_paths.append(" → ".join(path))
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(labels_test, predictions)
        
        # Detailed metrics
        metrics = self.classifier1._calculate_metrics(labels_test, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(labels_test, predictions, labels=list(range(len(self.config.data.class_names))))
        
        # Classification report
        class_report = classification_report(
            labels_test, predictions,
            target_names=self.config.data.class_names,
            output_dict=True,
            zero_division=0
        )
        
        self.logger.info(f"🎯 Overall Hierarchical Accuracy: {overall_accuracy:.4f}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'predictions': predictions,
            'true_labels': labels_test,
            'prediction_paths': prediction_paths,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
    
    def save_models(self, base_dir: str = None):
        """Save all trained models."""
        if base_dir is None:
            base_dir = self.config.classical_ml.models_dir
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Save individual classifiers
        if self.classifier1.is_fitted:
            self.classifier1.save_model(os.path.join(base_dir, 'classifier1.pkl'))
        
        if self.classifier2.is_fitted:
            self.classifier2.save_model(os.path.join(base_dir, 'classifier2.pkl'))
        
        if self.classifier3.is_fitted:
            self.classifier3.save_model(os.path.join(base_dir, 'classifier3.pkl'))
        
        # Save system metadata
        metadata = {
            'config': {
                'class_names': self.config.data.class_names,
                'healthy_idx': self.config.data.healthy_idx,
                'weed_idx': self.config.data.weed_idx,
                'disease_indices': self.config.data.disease_indices
            },
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results
        }
        
        metadata_path = os.path.join(base_dir, 'system_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"💾 All models saved to: {base_dir}")
    
    def load_models(self, base_dir: str = None):
        """Load all trained models."""
        if base_dir is None:
            base_dir = self.config.classical_ml.models_dir
        
        # Load individual classifiers
        classifier1_path = os.path.join(base_dir, 'classifier1.pkl')
        if os.path.exists(classifier1_path):
            self.classifier1.load_model(classifier1_path)
        
        classifier2_path = os.path.join(base_dir, 'classifier2.pkl')
        if os.path.exists(classifier2_path):
            self.classifier2.load_model(classifier2_path)
        
        classifier3_path = os.path.join(base_dir, 'classifier3.pkl')
        if os.path.exists(classifier3_path):
            self.classifier3.load_model(classifier3_path)
        
        # Load system metadata
        metadata_path = os.path.join(base_dir, 'system_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.training_results = metadata.get('training_results', {})
            self.evaluation_results = metadata.get('evaluation_results', {})
        
        self.logger.info(f"📂 Models loaded from: {base_dir}")
    
    def run_full_pipeline(self, hierarchical_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete classical ML pipeline.
        
        Args:
            hierarchical_splits: Hierarchical data splits
            
        Returns:
            Complete results including training and evaluation
        """
        self.logger.info("🚀 Running Complete Classical ML Pipeline")
        
        # Training
        training_results = self.train_hierarchical_system(hierarchical_splits)
        
        # Evaluation
        evaluation_results = self.evaluate_hierarchical_system()
        
        # Save models
        self.save_models()
        
        # Create visualizations
        self._create_visualizations()
        
        # Combine results
        complete_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'final_accuracy': evaluation_results.get('hierarchical', {}).get('overall_accuracy', 0.0),
            'feature_info': self.feature_extractor.get_feature_info()
        }
        
        self.logger.info(f"🎉 Classical ML Pipeline Completed!")
        if 'final_accuracy' in complete_results:
            self.logger.info(f"🎯 Final Accuracy: {complete_results['final_accuracy']:.4f}")
        
        return complete_results
    
    def _create_visualizations(self):
        """Create visualizations for classical ML results."""
        if not self.evaluation_results:
            return
        
        self.logger.info("📊 Creating visualizations...")
        
        # Confusion matrix for hierarchical system
        if 'hierarchical' in self.evaluation_results:
            hierarchical_results = self.evaluation_results['hierarchical']
            if 'confusion_matrix' in hierarchical_results:
                self._plot_confusion_matrix(
                    hierarchical_results['confusion_matrix'],
                    'Classical ML - Hierarchical System Confusion Matrix',
                    'classical_ml_confusion_matrix.png'
                )
        
        # Individual classifier performance
        self._plot_classifier_comparison()
        
        self.logger.info("✅ Visualizations created")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, title: str, filename: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.data.class_names,
                   yticklabels=self.config.data.class_names)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Plot saved: {plot_path}")
        
        plt.show()
    
    def _plot_classifier_comparison(self):
        """Plot comparison of individual classifiers."""
        if not self.evaluation_results:
            return
        
        accuracies = []
        classifier_names = []
        
        for classifier_name in ['classifier1', 'classifier2', 'classifier3']:
            if classifier_name in self.evaluation_results:
                results = self.evaluation_results[classifier_name]
                if 'metrics' in results:
                    accuracies.append(results['metrics']['accuracy'])
                    classifier_names.append(classifier_name.replace('classifier', 'Stage '))
        
        if hierarchical_acc := self.evaluation_results.get('hierarchical', {}).get('overall_accuracy'):
            accuracies.append(hierarchical_acc)
            classifier_names.append('Hierarchical System')
        
        if accuracies:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(classifier_names, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Classical ML - Classifier Performance Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy')
            plt.xlabel('Classifier')
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            if self.config.evaluation.save_plots:
                plot_path = os.path.join(self.config.evaluation.plots_dir, 'classical_ml_comparison.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"📁 Plot saved: {plot_path}")
            
            plt.show()
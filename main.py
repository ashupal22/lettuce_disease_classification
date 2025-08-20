#!/usr/bin/env python3
"""
Main execution script for Lettuce Disease Classification project.
Provides a unified entry point for all experiments and evaluations.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config, get_config
from src.utils.logger import create_session_logger, ExperimentLogger, LoggedOperation
from src.data_processing.data_exploration import DatasetExplorer
from src.data_processing.preprocessing import DataPreprocessingPipeline
from src.classical_ml.hierarchical_classifier import HierarchicalClassicalML
from src.deep_learning.hierarchical_cnn import HierarchicalCNN
from src.evaluation.evaluator import ComprehensiveEvaluator


class LettuceClassificationPipeline:
    """Main pipeline for lettuce disease classification experiments."""
    
    def __init__(self, config: Config):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = create_session_logger("lettuce_classification")
        self.experiment_logger = ExperimentLogger(self.logger, "Lettuce Disease Classification")
        
        # Initialize components
        self.explorer = DatasetExplorer(config)
        self.preprocessing_pipeline = DataPreprocessingPipeline(config)
        self.classical_ml = HierarchicalClassicalML(config)
        self.deep_learning = HierarchicalCNN(config)
        self.evaluator = ComprehensiveEvaluator(config)
        
        # Results storage
        self.results = {}
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline including all experiments.
        
        Returns:
            Dictionary containing all results
        """
        self.experiment_logger.start_experiment(
            "Complete lettuce disease classification pipeline with data exploration, "
            "classical ML baseline, and deep learning approach"
        )
        
        try:
            # 1. Data Exploration
            with LoggedOperation(self.logger, "Data Exploration"):
                exploration_results = self.explorer.explore_dataset()
                self.results['exploration'] = exploration_results
            
            # 2. Data Preprocessing
            with LoggedOperation(self.logger, "Data Preprocessing"):
                # Deep learning preprocessing
                dl_preprocessing = self.preprocessing_pipeline.run_deep_learning_preprocessing()
                self.results['dl_preprocessing'] = dl_preprocessing
                
                # Classical ML preprocessing
                classical_preprocessing = self.preprocessing_pipeline.run_classical_ml_preprocessing()
                self.results['classical_preprocessing'] = classical_preprocessing
            
            # 3. Classical ML Training and Evaluation
            with LoggedOperation(self.logger, "Classical ML Pipeline"):
                classical_results = self.classical_ml.run_full_pipeline(
                    classical_preprocessing['hierarchical_splits']
                )
                self.results['classical_ml'] = classical_results
            
            # 4. Deep Learning Training and Evaluation
            with LoggedOperation(self.logger, "Deep Learning Pipeline"):
                deep_learning_results = self.deep_learning.run_full_pipeline(
                    dl_preprocessing['data_loaders']
                )
                self.results['deep_learning'] = deep_learning_results
            
            # 5. Comprehensive Evaluation
            with LoggedOperation(self.logger, "Comprehensive Evaluation"):
                evaluation_results = self.evaluator.evaluate_all_approaches(
                    classical_results=classical_results,
                    deep_learning_results=deep_learning_results,
                    test_data=dl_preprocessing['data_loaders']['test_loader_full']
                )
                self.results['evaluation'] = evaluation_results
            
            # 6. Generate Final Report
            self._generate_final_report()
            
            self.experiment_logger.end_experiment(success=True, 
                                                summary="All experiments completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.experiment_logger.end_experiment(success=False, 
                                                summary=f"Pipeline failed: {e}")
            raise
        
        return self.results
    
    def run_exploration_only(self) -> Dict[str, Any]:
        """Run only data exploration."""
        self.experiment_logger.start_experiment("Data exploration and analysis")
        
        try:
            exploration_results = self.explorer.explore_dataset()
            self.results['exploration'] = exploration_results
            
            # Get preprocessing recommendations
            recommendations = self.explorer.get_preprocessing_recommendations()
            self.results['preprocessing_recommendations'] = recommendations
            
            self.experiment_logger.log_result("classes_found", 
                                             len(exploration_results['structure']['class_distribution']))
            self.experiment_logger.log_result("total_images", 
                                             exploration_results['structure']['total_images'])
            self.experiment_logger.log_result("imbalance_ratio", 
                                             exploration_results['structure'].get('imbalance_ratio', 1.0))
            
            self.experiment_logger.end_experiment(success=True)
            
        except Exception as e:
            self.logger.error(f"Exploration failed: {e}")
            self.experiment_logger.end_experiment(success=False, summary=f"Exploration failed: {e}")
            raise
        
        return self.results
    
    def run_classical_ml_only(self) -> Dict[str, Any]:
        """Run only classical ML pipeline."""
        self.experiment_logger.start_experiment("Classical ML baseline approach")
        
        try:
            # Preprocessing
            classical_preprocessing = self.preprocessing_pipeline.run_classical_ml_preprocessing()
            self.results['classical_preprocessing'] = classical_preprocessing
            
            # Classical ML training and evaluation
            classical_results = self.classical_ml.run_full_pipeline(
                classical_preprocessing['hierarchical_splits']
            )
            self.results['classical_ml'] = classical_results
            
            # Log key results
            if 'final_accuracy' in classical_results:
                self.experiment_logger.log_result("overall_accuracy", 
                                                 classical_results['final_accuracy'])
            
            self.experiment_logger.end_experiment(success=True)
            
        except Exception as e:
            self.logger.error(f"Classical ML pipeline failed: {e}")
            self.experiment_logger.end_experiment(success=False, 
                                                summary=f"Classical ML failed: {e}")
            raise
        
        return self.results
    
    def run_deep_learning_only(self) -> Dict[str, Any]:
        """Run only deep learning pipeline."""
        self.experiment_logger.start_experiment("Deep learning CNN approach")
        
        try:
            # Preprocessing
            dl_preprocessing = self.preprocessing_pipeline.run_deep_learning_preprocessing()
            self.results['dl_preprocessing'] = dl_preprocessing
            
            # Deep learning training and evaluation
            deep_learning_results = self.deep_learning.run_full_pipeline(
                dl_preprocessing['data_loaders']
            )
            self.results['deep_learning'] = deep_learning_results
            
            # Log key results
            if 'final_accuracy' in deep_learning_results:
                self.experiment_logger.log_result("overall_accuracy", 
                                                 deep_learning_results['final_accuracy'])
            
            self.experiment_logger.end_experiment(success=True)
            
        except Exception as e:
            self.logger.error(f"Deep learning pipeline failed: {e}")
            self.experiment_logger.end_experiment(success=False, 
                                                summary=f"Deep learning failed: {e}")
            raise
        
        return self.results
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        self.logger.info("📊 Generating final comprehensive report...")
        
        report_path = os.path.join(self.config.evaluation.reports_dir, "final_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("LETTUCE DISEASE CLASSIFICATION - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write("This report presents the results of a comprehensive lettuce disease\n")
            f.write("classification system using both classical machine learning and\n")
            f.write("deep learning approaches with a hierarchical classification strategy.\n\n")
            
            # Dataset Overview
            if 'exploration' in self.results:
                exploration = self.results['exploration']
                f.write("DATASET OVERVIEW\n")
                f.write("-" * 16 + "\n")
                f.write(f"Total Images: {exploration['structure']['total_images']:,}\n")
                f.write(f"Classes: {len(exploration['structure']['class_distribution'])}\n")
                f.write(f"Imbalance Ratio: {exploration['structure'].get('imbalance_ratio', 1.0):.1f}:1\n\n")
            
            # Results Comparison
            f.write("RESULTS COMPARISON\n")
            f.write("-" * 18 + "\n")
            
            classical_acc = None
            dl_acc = None
            
            if 'classical_ml' in self.results and 'final_accuracy' in self.results['classical_ml']:
                classical_acc = self.results['classical_ml']['final_accuracy']
                f.write(f"Classical ML Accuracy:     {classical_acc:.4f} ({classical_acc*100:.2f}%)\n")
            
            if 'deep_learning' in self.results and 'final_accuracy' in self.results['deep_learning']:
                dl_acc = self.results['deep_learning']['final_accuracy']
                f.write(f"Deep Learning Accuracy:    {dl_acc:.4f} ({dl_acc*100:.2f}%)\n")
            
            if classical_acc and dl_acc:
                improvement = ((dl_acc - classical_acc) / classical_acc) * 100
                f.write(f"Deep Learning Improvement: {improvement:+.1f}%\n")
            
            f.write("\n")
            
            # Methodology
            f.write("METHODOLOGY\n")
            f.write("-" * 11 + "\n")
            f.write("1. Data Exploration: Comprehensive analysis of dataset characteristics\n")
            f.write("2. Preprocessing: Image resizing, padding, and augmentation\n")
            f.write("3. Hierarchical Classification:\n")
            f.write("   - Stage 1: Crop vs Weed classification\n")
            f.write("   - Stage 2: Healthy vs Disease classification (crops only)\n")
            f.write("   - Stage 3: Disease type classification (diseases only)\n")
            f.write("4. Classical ML: Feature extraction + Random Forest/SVM\n")
            f.write("5. Deep Learning: CNN with weighted loss functions\n")
            f.write("6. Evaluation: Comprehensive metrics and comparisons\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if classical_acc and dl_acc:
                if dl_acc > classical_acc:
                    f.write("✅ Deep learning approach shows superior performance\n")
                    f.write("✅ Hierarchical strategy effectively handles class imbalance\n")
                else:
                    f.write("⚠️ Classical ML performs competitively\n")
                    f.write("⚠️ Consider ensemble methods combining both approaches\n")
            
            f.write("🔄 Continue data collection for minority disease classes\n")
            f.write("📊 Implement real-time monitoring system\n")
            f.write("🚀 Deploy best-performing model for production use\n\n")
            
            # Technical Details
            f.write("TECHNICAL IMPLEMENTATION\n")
            f.write("-" * 24 + "\n")
            f.write(f"Framework: PyTorch + scikit-learn\n")
            f.write(f"Image Size: {self.config.data.img_size}x{self.config.data.img_size}\n")
            f.write(f"Batch Size: {self.config.deep_learning.batch_size}\n")
            f.write(f"Epochs: {self.config.deep_learning.epochs}\n")
            f.write(f"Device: {self.config.deep_learning.device}\n")
        
        self.logger.info(f"📁 Final report saved to: {report_path}")
    
    def save_results(self, filepath: Optional[str] = None):
        """Save all results to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.config.evaluation.results_dir, "complete_results.json")
        
        # Convert any non-serializable objects
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to: {filepath}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):  # custom objects
            return str(obj)
        else:
            return obj


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lettuce Disease Classification Pipeline")
    
    parser.add_argument('--mode', choices=['full', 'explore', 'classical', 'deep_learning'], 
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, help='Path to output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = get_config(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data.raw_data_dir = args.data_dir
    if args.output_dir:
        config.evaluation.results_dir = args.output_dir
        config.evaluation.plots_dir = os.path.join(args.output_dir, 'plots')
        config.evaluation.reports_dir = os.path.join(args.output_dir, 'reports')
        config.evaluation.metrics_dir = os.path.join(args.output_dir, 'metrics')
    if args.epochs:
        config.deep_learning.epochs = args.epochs
    if args.batch_size:
        config.deep_learning.batch_size = args.batch_size
    if args.device:
        config.deep_learning.device = args.device
    
    # Set logging level
    if args.verbose:
        config.logging.level = "DEBUG"
    
    # Validate configuration
    if not config.validate_paths():
        print("❌ Configuration validation failed. Please check your dataset path.")
        sys.exit(1)
    
    # Save configuration
    config.save_config()
    
    # Print configuration
    print(config)
    
    # Initialize and run pipeline
    pipeline = LettuceClassificationPipeline(config)
    
    try:
        # Run selected mode
        if args.mode == 'full':
            results = pipeline.run_full_pipeline()
        elif args.mode == 'explore':
            results = pipeline.run_exploration_only()
        elif args.mode == 'classical':
            results = pipeline.run_classical_ml_only()
        elif args.mode == 'deep_learning':
            results = pipeline.run_deep_learning_only()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Save results
        pipeline.save_results()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Results saved to: {config.evaluation.results_dir}")
        print(f"📊 Plots saved to: {config.evaluation.plots_dir}")
        print(f"📋 Reports saved to: {config.evaluation.reports_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
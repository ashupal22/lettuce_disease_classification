"""
Comprehensive evaluator for comparing classical ML and deep learning approaches.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..utils.config import Config
from ..utils.logger import get_logger


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all approaches."""
    
    def __init__(self, config: Config):
        """
        Initialize comprehensive evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
    def evaluate_all_approaches(self,
                               classical_results: Dict[str, Any],
                               deep_learning_results: Dict[str, Any],
                               test_data: Any = None) -> Dict[str, Any]:
        """
        Evaluate and compare all approaches.
        
        Args:
            classical_results: Results from classical ML
            deep_learning_results: Results from deep learning
            test_data: Test data (optional)
            
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("📊 Starting comprehensive evaluation...")
        
        evaluation_results = {
            'classical_ml': self._extract_classical_metrics(classical_results),
            'deep_learning': self._extract_deep_learning_metrics(deep_learning_results),
            'comparison': {}
        }
        
        # Compare approaches
        comparison = self._compare_approaches(
            evaluation_results['classical_ml'],
            evaluation_results['deep_learning']
        )
        evaluation_results['comparison'] = comparison
        
        # Create comprehensive visualizations
        self._create_comprehensive_visualizations(evaluation_results)
        
        # Generate comparison report
        self._generate_comparison_report(evaluation_results)
        
        # Save evaluation results
        self._save_evaluation_results(evaluation_results)
        
        self.logger.info("✅ Comprehensive evaluation completed")
        
        return evaluation_results
    
    def _extract_classical_metrics(self, classical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from classical ML results."""
        metrics = {}
        
        if 'evaluation_results' in classical_results:
            eval_results = classical_results['evaluation_results']
            
            # Individual classifier metrics
            for classifier_name in ['classifier1', 'classifier2', 'classifier3']:
                if classifier_name in eval_results:
                    classifier_results = eval_results[classifier_name]
                    if 'metrics' in classifier_results:
                        metrics[classifier_name] = classifier_results['metrics']
            
            # Hierarchical system metrics
            if 'hierarchical' in eval_results:
                hierarchical = eval_results['hierarchical']
                metrics['hierarchical'] = {
                    'overall_accuracy': hierarchical.get('overall_accuracy', 0.0),
                    'confusion_matrix': hierarchical.get('confusion_matrix'),
                    'classification_report': hierarchical.get('classification_report')
                }
        
        # Final accuracy
        metrics['final_accuracy'] = classical_results.get('final_accuracy', 0.0)
        
        return metrics
    
    def _extract_deep_learning_metrics(self, deep_learning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from deep learning results."""
        metrics = {}
        
        if 'evaluation_results' in deep_learning_results:
            eval_results = deep_learning_results['evaluation_results']
            
            # Individual model metrics
            for model_name in ['model1', 'model2', 'model3']:
                if model_name in eval_results:
                    model_results = eval_results[model_name]
                    if 'accuracy' in model_results:
                        metrics[model_name] = {
                            'accuracy': model_results['accuracy'],
                            'confusion_matrix': model_results.get('confusion_matrix'),
                            'classification_report': model_results.get('classification_report')
                        }
            
            # Hierarchical system metrics
            if 'hierarchical' in eval_results:
                hierarchical = eval_results['hierarchical']
                metrics['hierarchical'] = {
                    'overall_accuracy': hierarchical.get('overall_accuracy', 0.0),
                    'confusion_matrix': hierarchical.get('confusion_matrix'),
                    'classification_report': hierarchical.get('classification_report')
                }
        
        # Final accuracy
        metrics['final_accuracy'] = deep_learning_results.get('final_accuracy', 0.0)
        
        return metrics
    
    def _compare_approaches(self, 
                           classical_metrics: Dict[str, Any],
                           deep_learning_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare classical ML and deep learning approaches."""
        
        comparison = {}
        
        # Overall accuracy comparison
        classical_acc = classical_metrics.get('final_accuracy', 0.0)
        dl_acc = deep_learning_metrics.get('final_accuracy', 0.0)
        
        comparison['overall_accuracy'] = {
            'classical_ml': classical_acc,
            'deep_learning': dl_acc,
            'improvement': dl_acc - classical_acc,
            'improvement_percentage': ((dl_acc - classical_acc) / classical_acc * 100) if classical_acc > 0 else 0.0,
            'winner': 'deep_learning' if dl_acc > classical_acc else 'classical_ml'
        }
        
        # Stage-wise comparison
        stage_comparison = {}
        
        # Map classifier/model names
        stage_mapping = {
            'classifier1': 'model1',  # Crop vs Weed
            'classifier2': 'model2',  # Healthy vs Disease
            'classifier3': 'model3'   # Disease Classification
        }
        
        for classical_name, dl_name in stage_mapping.items():
            if classical_name in classical_metrics and dl_name in deep_learning_metrics:
                classical_stage = classical_metrics[classical_name]
                dl_stage = deep_learning_metrics[dl_name]
                
                classical_stage_acc = classical_stage.get('accuracy', 0.0)
                dl_stage_acc = dl_stage.get('accuracy', 0.0)
                
                stage_comparison[classical_name] = {
                    'classical_ml': classical_stage_acc,
                    'deep_learning': dl_stage_acc,
                    'improvement': dl_stage_acc - classical_stage_acc,
                    'winner': 'deep_learning' if dl_stage_acc > classical_stage_acc else 'classical_ml'
                }
        
        comparison['stage_wise'] = stage_comparison
        
        # Summary statistics
        comparison['summary'] = {
            'stages_won_by_classical': sum(1 for stage in stage_comparison.values() if stage['winner'] == 'classical_ml'),
            'stages_won_by_deep_learning': sum(1 for stage in stage_comparison.values() if stage['winner'] == 'deep_learning'),
            'average_improvement': np.mean([stage['improvement'] for stage in stage_comparison.values()]),
            'best_performing_approach': comparison['overall_accuracy']['winner']
        }
        
        return comparison
    
    def _create_comprehensive_visualizations(self, evaluation_results: Dict[str, Any]):
        """Create comprehensive comparison visualizations."""
        self.logger.info("📊 Creating comprehensive visualizations...")
        
        # Performance comparison plot
        self._plot_performance_comparison(evaluation_results)
        
        # Stage-wise comparison
        self._plot_stage_wise_comparison(evaluation_results)
        
        # Confusion matrix comparison
        self._plot_confusion_matrix_comparison(evaluation_results)
        
        self.logger.info("✅ Comprehensive visualizations created")
    
    def _plot_performance_comparison(self, evaluation_results: Dict[str, Any]):
        """Plot overall performance comparison."""
        comparison = evaluation_results.get('comparison', {})
        
        if 'overall_accuracy' not in comparison:
            return
        
        overall_acc = comparison['overall_accuracy']
        
        # Data for plotting
        approaches = ['Classical ML', 'Deep Learning']
        accuracies = [overall_acc['classical_ml'], overall_acc['deep_learning']]
        colors = ['skyblue', 'lightcoral']
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(approaches, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add improvement annotation
        improvement = overall_acc['improvement']
        improvement_pct = overall_acc['improvement_percentage']
        
        if improvement > 0:
            plt.annotate(f'Improvement: +{improvement:.4f}\n({improvement_pct:+.1f}%)',
                        xy=(1, accuracies[1]), xytext=(1.3, accuracies[1]),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=10, ha='center', color='green', fontweight='bold')
        elif improvement < 0:
            plt.annotate(f'Difference: {improvement:.4f}\n({improvement_pct:.1f}%)',
                        xy=(0, accuracies[0]), xytext=(-0.3, accuracies[0]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, ha='center', color='red', fontweight='bold')
        
        plt.title('Overall Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Approach', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add winner annotation
        winner = overall_acc['winner'].replace('_', ' ').title()
        plt.text(0.5, 0.95, f'Winner: {winner}', transform=plt.gca().transAxes,
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'performance_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Plot saved: {plot_path}")
        
        plt.show()
    
    def _plot_stage_wise_comparison(self, evaluation_results: Dict[str, Any]):
        """Plot stage-wise performance comparison."""
        comparison = evaluation_results.get('comparison', {})
        
        if 'stage_wise' not in comparison:
            return
        
        stage_wise = comparison['stage_wise']
        
        # Prepare data
        stages = []
        classical_accs = []
        dl_accs = []
        
        stage_names = {
            'classifier1': 'Stage 1: Crop vs Weed',
            'classifier2': 'Stage 2: Healthy vs Disease', 
            'classifier3': 'Stage 3: Disease Classification'
        }
        
        for stage_key, stage_data in stage_wise.items():
            if stage_key in stage_names:
                stages.append(stage_names[stage_key])
                classical_accs.append(stage_data['classical_ml'])
                dl_accs.append(stage_data['deep_learning'])
        
        if not stages:
            return
        
        # Create grouped bar plot
        x = np.arange(len(stages))
        width = 0.35
        
        plt.figure(figsize=(12, 7))
        
        bars1 = plt.bar(x - width/2, classical_accs, width, label='Classical ML', 
                       color='skyblue', alpha=0.8, edgecolor='black')
        bars2 = plt.bar(x + width/2, dl_accs, width, label='Deep Learning',
                       color='lightcoral', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.title('Stage-wise Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Classification Stage', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(x, stages, rotation=15, ha='right')
        plt.legend(fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'stage_wise_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Plot saved: {plot_path}")
        
        plt.show()
    
    def _plot_confusion_matrix_comparison(self, evaluation_results: Dict[str, Any]):
        """Plot confusion matrices side by side for comparison."""
        classical_metrics = evaluation_results.get('classical_ml', {})
        dl_metrics = evaluation_results.get('deep_learning', {})
        
        # Get hierarchical confusion matrices
        classical_cm = classical_metrics.get('hierarchical', {}).get('confusion_matrix')
        dl_cm = dl_metrics.get('hierarchical', {}).get('confusion_matrix')
        
        if classical_cm is None or dl_cm is None:
            return
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Classical ML confusion matrix
        sns.heatmap(classical_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.data.class_names,
                   yticklabels=self.config.data.class_names,
                   ax=ax1)
        ax1.set_title('Classical ML - Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Deep Learning confusion matrix
        sns.heatmap(dl_cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=self.config.data.class_names,
                   yticklabels=self.config.data.class_names,
                   ax=ax2)
        ax2.set_title('Deep Learning - Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        # Rotate labels
        for ax in [ax1, ax2]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'confusion_matrix_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Plot saved: {plot_path}")
        
        plt.show()
    
    def _generate_comparison_report(self, evaluation_results: Dict[str, Any]):
        """Generate detailed comparison report."""
        self.logger.info("📝 Generating comparison report...")
        
        report_path = os.path.join(self.config.evaluation.reports_dir, 'approach_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("LETTUCE DISEASE CLASSIFICATION - APPROACH COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 17 + "\n")
            
            comparison = evaluation_results.get('comparison', {})
            if 'overall_accuracy' in comparison:
                overall = comparison['overall_accuracy']
                winner = overall['winner'].replace('_', ' ').title()
                
                f.write(f"Best Performing Approach: {winner}\n")
                f.write(f"Classical ML Accuracy: {overall['classical_ml']:.4f} ({overall['classical_ml']*100:.2f}%)\n")
                f.write(f"Deep Learning Accuracy: {overall['deep_learning']:.4f} ({overall['deep_learning']*100:.2f}%)\n")
                f.write(f"Performance Difference: {overall['improvement']:+.4f} ({overall['improvement_percentage']:+.1f}%)\n\n")
            
            # Detailed Analysis
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 17 + "\n")
            
            if 'stage_wise' in comparison:
                f.write("Stage-wise Performance:\n")
                stage_wise = comparison['stage_wise']
                
                stage_names = {
                    'classifier1': 'Stage 1 (Crop vs Weed)',
                    'classifier2': 'Stage 2 (Healthy vs Disease)',
                    'classifier3': 'Stage 3 (Disease Classification)'
                }
                
                for stage_key, stage_data in stage_wise.items():
                    if stage_key in stage_names:
                        stage_name = stage_names[stage_key]
                        winner = stage_data['winner'].replace('_', ' ').title()
                        
                        f.write(f"\n{stage_name}:\n")
                        f.write(f"  Classical ML: {stage_data['classical_ml']:.4f}\n")
                        f.write(f"  Deep Learning: {stage_data['deep_learning']:.4f}\n")
                        f.write(f"  Improvement: {stage_data['improvement']:+.4f}\n")
                        f.write(f"  Winner: {winner}\n")
            
            # Summary Statistics
            if 'summary' in comparison:
                summary = comparison['summary']
                f.write(f"\nSUMMARY STATISTICS\n")
                f.write("-" * 18 + "\n")
                f.write(f"Stages won by Classical ML: {summary['stages_won_by_classical']}\n")
                f.write(f"Stages won by Deep Learning: {summary['stages_won_by_deep_learning']}\n")
                f.write(f"Average improvement per stage: {summary['average_improvement']:+.4f}\n")
                f.write(f"Overall best approach: {summary['best_performing_approach'].replace('_', ' ').title()}\n")
            
            # Methodology Notes
            f.write(f"\nMETHODOLOGY\n")
            f.write("-" * 11 + "\n")
            f.write("Both approaches used identical hierarchical classification strategy:\n")
            f.write("1. Stage 1: Crop vs Weed classification\n")
            f.write("2. Stage 2: Healthy vs Disease classification (crops only)\n")
            f.write("3. Stage 3: Disease type classification (diseases only)\n\n")
            f.write("Classical ML: Feature extraction + Random Forest/SVM\n")
            f.write("Deep Learning: CNN with weighted asymmetric loss\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if 'overall_accuracy' in comparison:
                overall = comparison['overall_accuracy']
                if overall['deep_learning'] > overall['classical_ml']:
                    f.write("✅ Deep Learning approach recommended for deployment\n")
                    f.write("✅ Shows superior performance across most stages\n")
                    f.write("✅ Better handling of complex visual patterns\n")
                    f.write("⚠️ Requires more computational resources\n")
                    f.write("⚠️ Less interpretable than classical features\n")
                else:
                    f.write("✅ Classical ML approach shows competitive performance\n")
                    f.write("✅ Faster inference and lower resource requirements\n")
                    f.write("✅ More interpretable feature analysis\n")
                    f.write("🔄 Consider ensemble approach combining both methods\n")
            
            f.write("\n🔄 Consider hybrid approach for optimal results\n")
            f.write("📊 Continue data collection for minority classes\n")
            f.write("🚀 Implement real-time monitoring system\n")
        
        self.logger.info(f"📁 Comparison report saved: {report_path}")
    
    def _save_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Save evaluation results to JSON file."""
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(evaluation_results)
        
        results_path = os.path.join(self.config.evaluation.results_dir, 'comprehensive_evaluation.json')
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Evaluation results saved: {results_path}")
        
        # Also save as CSV for easy analysis
        self._save_comparison_csv(evaluation_results)
    
    def _save_comparison_csv(self, evaluation_results: Dict[str, Any]):
        """Save comparison results as CSV."""
        
        comparison_data = []
        
        # Overall comparison
        if 'comparison' in evaluation_results and 'overall_accuracy' in evaluation_results['comparison']:
            overall = evaluation_results['comparison']['overall_accuracy']
            comparison_data.append({
                'Stage': 'Overall System',
                'Classical_ML_Accuracy': overall['classical_ml'],
                'Deep_Learning_Accuracy': overall['deep_learning'],
                'Improvement': overall['improvement'],
                'Improvement_Percentage': overall['improvement_percentage'],
                'Winner': overall['winner']
            })
        
        # Stage-wise comparison
        if 'comparison' in evaluation_results and 'stage_wise' in evaluation_results['comparison']:
            stage_wise = evaluation_results['comparison']['stage_wise']
            
            stage_names = {
                'classifier1': 'Stage 1: Crop vs Weed',
                'classifier2': 'Stage 2: Healthy vs Disease',
                'classifier3': 'Stage 3: Disease Classification'
            }
            
            for stage_key, stage_data in stage_wise.items():
                if stage_key in stage_names:
                    comparison_data.append({
                        'Stage': stage_names[stage_key],
                        'Classical_ML_Accuracy': stage_data['classical_ml'],
                        'Deep_Learning_Accuracy': stage_data['deep_learning'],
                        'Improvement': stage_data['improvement'],
                        'Improvement_Percentage': (stage_data['improvement'] / stage_data['classical_ml'] * 100) if stage_data['classical_ml'] > 0 else 0,
                        'Winner': stage_data['winner']
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            csv_path = os.path.join(self.config.evaluation.results_dir, 'approach_comparison.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"📁 Comparison CSV saved: {csv_path}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
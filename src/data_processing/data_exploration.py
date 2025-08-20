"""
Data exploration module for lettuce disease classification.
Provides comprehensive analysis of the dataset characteristics.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import Config
from ..utils.logger import get_logger


class DatasetExplorer:
    """Dataset exploration and analysis class."""
    
    def __init__(self, config: Config):
        """
        Initialize dataset explorer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.dataset_stats = {}
        self.image_stats = defaultdict(list)
        
    def explore_dataset(self) -> Dict[str, Any]:
        """
        Perform comprehensive dataset exploration.
        
        Returns:
            Dictionary containing exploration results
        """
        self.logger.info("🔍 Starting comprehensive dataset exploration...")
        
        # Check dataset structure
        self._analyze_dataset_structure()
        
        # Analyze images
        self._analyze_images()
        
        # Generate statistics
        self._generate_statistics()
        
        # Create visualizations
        self._create_visualizations()
        
        # Save exploration results
        self._save_results()
        
        self.logger.info("✅ Dataset exploration completed successfully!")
        return self.dataset_stats
    
    def _analyze_dataset_structure(self):
        """Analyze the basic structure of the dataset."""
        self.logger.info("📁 Analyzing dataset structure...")
        
        raw_data_path = self.config.data.raw_data_dir
        
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Dataset directory not found: {raw_data_path}")
        
        # Count files per class
        class_info = {}
        total_images = 0
        
        for class_name in self.config.data.class_names:
            class_dir = os.path.join(raw_data_path, class_name)
            
            if os.path.exists(class_dir):
                # Count image files
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                
                class_info[class_name] = {
                    'count': len(image_files),
                    'files': image_files[:10],  # Store first 10 filenames as examples
                    'directory_exists': True
                }
                total_images += len(image_files)
                
                self.logger.info(f"  📂 {class_name}: {len(image_files)} images")
            else:
                class_info[class_name] = {
                    'count': 0,
                    'files': [],
                    'directory_exists': False
                }
                self.logger.warning(f"  ⚠️ {class_name}: Directory not found")
        
        self.dataset_stats['structure'] = {
            'total_classes': len(self.config.data.class_names),
            'total_images': total_images,
            'class_info': class_info,
            'class_distribution': {name: info['count'] for name, info in class_info.items()},
            'healthy_images': class_info[self.config.data.healthy_class]['count'],
            'weed_images': class_info[self.config.data.weed_class]['count'],
            'disease_images': sum(class_info[name]['count'] for name in self.config.data.disease_names)
        }
        
        # Calculate class imbalance
        counts = [info['count'] for info in class_info.values() if info['count'] > 0]
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            self.dataset_stats['structure']['imbalance_ratio'] = imbalance_ratio
            
            if imbalance_ratio > 10:
                self.logger.warning(f"⚖️ Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            elif imbalance_ratio > 5:
                self.logger.info(f"⚖️ Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
    
    def _analyze_images(self):
        """Analyze image characteristics across all classes."""
        self.logger.info("🖼️ Analyzing image characteristics...")
        
        raw_data_path = self.config.data.raw_data_dir
        sample_size = 50  # Analyze 50 images per class for statistics
        
        all_image_stats = {
            'widths': [],
            'heights': [],
            'aspect_ratios': [],
            'file_sizes': [],
            'modes': [],
            'formats': []
        }
        
        class_image_stats = {}
        
        for class_name in self.config.data.class_names:
            class_dir = os.path.join(raw_data_path, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            class_stats = {
                'widths': [],
                'heights': [],
                'aspect_ratios': [],
                'file_sizes': [],
                'modes': [],
                'formats': []
            }
            
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # Sample images for analysis
            sample_files = np.random.choice(image_files, 
                                          min(sample_size, len(image_files)), 
                                          replace=False) if image_files else []
            
            for img_file in sample_files:
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    # Get file size
                    file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
                    
                    # Open image and get properties
                    with Image.open(img_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height
                        mode = img.mode
                        format_type = img.format or 'Unknown'
                    
                    # Store statistics
                    stats_to_update = [class_stats, all_image_stats]
                    for stats in stats_to_update:
                        stats['widths'].append(width)
                        stats['heights'].append(height)
                        stats['aspect_ratios'].append(aspect_ratio)
                        stats['file_sizes'].append(file_size)
                        stats['modes'].append(mode)
                        stats['formats'].append(format_type)
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {img_path}: {e}")
            
            # Calculate class statistics
            if class_stats['widths']:  # If we have data
                class_image_stats[class_name] = {
                    'sample_size': len(class_stats['widths']),
                    'width_stats': self._calculate_stats(class_stats['widths']),
                    'height_stats': self._calculate_stats(class_stats['heights']),
                    'aspect_ratio_stats': self._calculate_stats(class_stats['aspect_ratios']),
                    'file_size_stats': self._calculate_stats(class_stats['file_sizes']),
                    'most_common_mode': Counter(class_stats['modes']).most_common(1)[0][0] if class_stats['modes'] else 'Unknown',
                    'most_common_format': Counter(class_stats['formats']).most_common(1)[0][0] if class_stats['formats'] else 'Unknown'
                }
        
        # Calculate overall statistics
        if all_image_stats['widths']:
            self.dataset_stats['images'] = {
                'total_analyzed': len(all_image_stats['widths']),
                'width_stats': self._calculate_stats(all_image_stats['widths']),
                'height_stats': self._calculate_stats(all_image_stats['heights']),
                'aspect_ratio_stats': self._calculate_stats(all_image_stats['aspect_ratios']),
                'file_size_stats': self._calculate_stats(all_image_stats['file_sizes']),
                'mode_distribution': dict(Counter(all_image_stats['modes'])),
                'format_distribution': dict(Counter(all_image_stats['formats'])),
                'class_specific': class_image_stats
            }
            
            # Log key findings
            self.logger.info(f"  📊 Analyzed {len(all_image_stats['widths'])} images")
            self.logger.info(f"  📏 Image size range: {min(all_image_stats['widths'])}x{min(all_image_stats['heights'])} to {max(all_image_stats['widths'])}x{max(all_image_stats['heights'])}")
            self.logger.info(f"  📁 File size range: {min(all_image_stats['file_sizes']):.2f} MB to {max(all_image_stats['file_sizes']):.2f} MB")
            self.logger.info(f"  🎨 Most common mode: {Counter(all_image_stats['modes']).most_common(1)[0][0]}")
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {}
        
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
    
    def _generate_statistics(self):
        """Generate comprehensive dataset statistics."""
        self.logger.info("📈 Generating dataset statistics...")
        
        if 'structure' not in self.dataset_stats:
            self.logger.warning("No structure data available for statistics generation")
            return
        
        structure = self.dataset_stats['structure']
        class_counts = structure['class_distribution']
        
        # Basic statistics
        total_images = structure['total_images']
        non_zero_counts = [count for count in class_counts.values() if count > 0]
        
        stats = {
            'dataset_size': total_images,
            'num_classes': len([count for count in class_counts.values() if count > 0]),
            'class_balance': {
                'most_frequent_class': max(class_counts, key=class_counts.get),
                'least_frequent_class': min(class_counts, key=class_counts.get),
                'imbalance_ratio': structure.get('imbalance_ratio', 1.0),
                'class_percentages': {name: (count/total_images)*100 
                                   for name, count in class_counts.items() if count > 0}
            },
            'hierarchical_distribution': {
                'healthy_percentage': (structure['healthy_images']/total_images)*100 if total_images > 0 else 0,
                'disease_percentage': (structure['disease_images']/total_images)*100 if total_images > 0 else 0,
                'weed_percentage': (structure['weed_images']/total_images)*100 if total_images > 0 else 0
            }
        }
        
        if non_zero_counts:
            stats['class_statistics'] = {
                'mean_samples_per_class': np.mean(non_zero_counts),
                'std_samples_per_class': np.std(non_zero_counts),
                'min_samples': min(non_zero_counts),
                'max_samples': max(non_zero_counts)
            }
        
        self.dataset_stats['statistics'] = stats
        
        # Log key statistics
        self.logger.info(f"  📊 Total dataset size: {total_images:,} images")
        self.logger.info(f"  🏷️ Active classes: {stats['num_classes']}/{len(self.config.data.class_names)}")
        self.logger.info(f"  ⚖️ Class imbalance ratio: {structure.get('imbalance_ratio', 1.0):.1f}:1")
        self.logger.info(f"  🌱 Healthy: {stats['hierarchical_distribution']['healthy_percentage']:.1f}%")
        self.logger.info(f"  🦠 Disease: {stats['hierarchical_distribution']['disease_percentage']:.1f}%")
        self.logger.info(f"  🌿 Weed: {stats['hierarchical_distribution']['weed_percentage']:.1f}%")
    
    def _create_visualizations(self):
        """Create comprehensive visualizations of the dataset."""
        self.logger.info("📊 Creating dataset visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Class distribution bar plot
        ax1 = plt.subplot(2, 3, 1)
        self._plot_class_distribution(ax1)
        
        # 2. Hierarchical distribution pie chart
        ax2 = plt.subplot(2, 3, 2)
        self._plot_hierarchical_distribution(ax2)
        
        # 3. Image size distribution
        ax3 = plt.subplot(2, 3, 3)
        self._plot_image_size_distribution(ax3)
        
        # 4. File size distribution
        ax4 = plt.subplot(2, 3, 4)
        self._plot_file_size_distribution(ax4)
        
        # 5. Aspect ratio distribution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_aspect_ratio_distribution(ax5)
        
        # 6. Class imbalance heatmap
        ax6 = plt.subplot(2, 3, 6)
        self._plot_class_imbalance_heatmap(ax6)
        
        plt.suptitle('Lettuce Disease Dataset - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'dataset_exploration.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Visualization saved to: {plot_path}")
        
        plt.show()
        
        # Create sample images visualization
        self._create_sample_images_visualization()
    
    def _plot_class_distribution(self, ax):
        """Plot class distribution bar chart."""
        if 'structure' not in self.dataset_stats:
            return
        
        class_counts = self.dataset_stats['structure']['class_distribution']
        
        # Filter out zero counts
        filtered_counts = {k: v for k, v in class_counts.items() if v > 0}
        
        classes = list(filtered_counts.keys())
        counts = list(filtered_counts.values())
        
        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        bars = ax.bar(range(len(classes)), counts, color=colors)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Images')
        ax.set_title('Class Distribution')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                   f'{count}', ha='center', va='bottom', fontsize=9)
    
    def _plot_hierarchical_distribution(self, ax):
        """Plot hierarchical distribution pie chart."""
        if 'structure' not in self.dataset_stats:
            return
        
        structure = self.dataset_stats['structure']
        
        labels = ['Healthy', 'Disease', 'Weed']
        sizes = [
            structure['healthy_images'],
            structure['disease_images'],
            structure['weed_images']
        ]
        
        # Filter out zero values
        filtered_data = [(label, size) for label, size in zip(labels, sizes) if size > 0]
        if not filtered_data:
            return
        
        labels, sizes = zip(*filtered_data)
        colors = ['#90EE90', '#FF6B6B', '#FFD700'][:len(sizes)]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, explode=[0.05]*len(sizes))
        ax.set_title('Hierarchical Distribution')
        
        # Beautify text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_image_size_distribution(self, ax):
        """Plot image size distribution."""
        if 'images' not in self.dataset_stats:
            ax.text(0.5, 0.5, 'No image data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Image Size Distribution')
            return
        
        images_data = self.dataset_stats['images']
        width_stats = images_data['width_stats']
        height_stats = images_data['height_stats']
        
        # Create scatter plot of width vs height
        if 'class_specific' in images_data:
            for i, (class_name, class_data) in enumerate(images_data['class_specific'].items()):
                if 'width_stats' in class_data and 'height_stats' in class_data:
                    ax.scatter(class_data['width_stats']['mean'], 
                             class_data['height_stats']['mean'],
                             label=class_name, s=100, alpha=0.7)
        
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.set_title('Average Image Dimensions by Class')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_file_size_distribution(self, ax):
        """Plot file size distribution."""
        if 'images' not in self.dataset_stats:
            ax.text(0.5, 0.5, 'No image data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('File Size Distribution')
            return
        
        images_data = self.dataset_stats['images']
        
        if 'class_specific' in images_data:
            file_sizes = []
            class_labels = []
            
            for class_name, class_data in images_data['class_specific'].items():
                if 'file_size_stats' in class_data:
                    file_sizes.append(class_data['file_size_stats']['mean'])
                    class_labels.append(class_name)
            
            if file_sizes:
                colors = plt.cm.viridis(np.linspace(0, 1, len(file_sizes)))
                bars = ax.bar(range(len(file_sizes)), file_sizes, color=colors)
                ax.set_xlabel('Classes')
                ax.set_ylabel('Average File Size (MB)')
                ax.set_title('Average File Size by Class')
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels, rotation=45, ha='right')
                
                # Add value labels
                for bar, size in zip(bars, file_sizes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(file_sizes)*0.01,
                           f'{size:.2f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_aspect_ratio_distribution(self, ax):
        """Plot aspect ratio distribution."""
        if 'images' not in self.dataset_stats:
            ax.text(0.5, 0.5, 'No image data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Aspect Ratio Distribution')
            return
        
        images_data = self.dataset_stats['images']
        
        if 'class_specific' in images_data:
            aspect_ratios = []
            class_labels = []
            
            for class_name, class_data in images_data['class_specific'].items():
                if 'aspect_ratio_stats' in class_data:
                    aspect_ratios.append(class_data['aspect_ratio_stats']['mean'])
                    class_labels.append(class_name)
            
            if aspect_ratios:
                ax.scatter(range(len(aspect_ratios)), aspect_ratios, s=100, alpha=0.7)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Square (1:1)')
                ax.set_xlabel('Classes')
                ax.set_ylabel('Average Aspect Ratio')
                ax.set_title('Average Aspect Ratio by Class')
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    def _plot_class_imbalance_heatmap(self, ax):
        """Plot class imbalance as a heatmap."""
        if 'structure' not in self.dataset_stats:
            return
        
        class_counts = self.dataset_stats['structure']['class_distribution']
        
        # Create imbalance matrix
        classes = [k for k, v in class_counts.items() if v > 0]
        counts = [class_counts[k] for k in classes]
        
        if len(classes) < 2:
            ax.text(0.5, 0.5, 'Insufficient classes for imbalance analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Class Imbalance Analysis')
            return
        
        # Calculate imbalance ratios
        imbalance_matrix = np.zeros((len(classes), len(classes)))
        for i, count_i in enumerate(counts):
            for j, count_j in enumerate(counts):
                if count_j > 0:
                    imbalance_matrix[i, j] = count_i / count_j
        
        # Create heatmap
        im = ax.imshow(imbalance_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_title('Class Imbalance Ratios')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Ratio')
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{imbalance_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black" if imbalance_matrix[i, j] < 5 else "white",
                             fontsize=8)
    
    def _create_sample_images_visualization(self):
        """Create a grid showing sample images from each class."""
        self.logger.info("🖼️ Creating sample images visualization...")
        
        raw_data_path = self.config.data.raw_data_dir
        samples_per_class = 3
        
        # Calculate grid size
        num_classes = len([name for name in self.config.data.class_names 
                          if os.path.exists(os.path.join(raw_data_path, name))])
        
        if num_classes == 0:
            self.logger.warning("No valid class directories found for sample visualization")
            return
        
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                               figsize=(15, 3*num_classes))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        row = 0
        for class_name in self.config.data.class_names:
            class_dir = os.path.join(raw_data_path, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                for col in range(samples_per_class):
                    axes[row, col].text(0.5, 0.5, 'No images', ha='center', va='center')
                    axes[row, col].set_title(f'{class_name} - No Data')
                    axes[row, col].axis('off')
                row += 1
                continue
            
            # Sample random images
            sample_files = np.random.choice(image_files, 
                                          min(samples_per_class, len(image_files)), 
                                          replace=False)
            
            for col, img_file in enumerate(sample_files):
                img_path = os.path.join(class_dir, img_file)
                
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f'{class_name}\n{img.size[0]}x{img.size[1]}', fontsize=10)
                    axes[row, col].axis('off')
                    
                except Exception as e:
                    axes[row, col].text(0.5, 0.5, f'Error loading\n{str(e)[:20]}...', 
                                      ha='center', va='center')
                    axes[row, col].set_title(f'{class_name} - Error')
                    axes[row, col].axis('off')
            
            # Fill empty columns if fewer samples than requested
            for col in range(len(sample_files), samples_per_class):
                axes[row, col].axis('off')
            
            row += 1
        
        plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if self.config.evaluation.save_plots:
            plot_path = os.path.join(self.config.evaluation.plots_dir, 'sample_images.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 Sample images visualization saved to: {plot_path}")
        
        plt.show()
    
    def _save_results(self):
        """Save exploration results to files."""
        self.logger.info("💾 Saving exploration results...")
        
        # Save as JSON
        results_path = os.path.join(self.config.evaluation.results_dir, 'dataset_exploration.json')
        with open(results_path, 'w') as f:
            json.dump(self.dataset_stats, f, indent=2, default=str)
        
        # Save as detailed report
        report_path = os.path.join(self.config.evaluation.reports_dir, 'dataset_exploration_report.txt')
        self._generate_text_report(report_path)
        
        # Save as CSV for easy analysis
        csv_path = os.path.join(self.config.evaluation.results_dir, 'class_distribution.csv')
        self._save_class_distribution_csv(csv_path)
        
        self.logger.info(f"📁 Results saved to:")
        self.logger.info(f"  - JSON: {results_path}")
        self.logger.info(f"  - Report: {report_path}")
        self.logger.info(f"  - CSV: {csv_path}")
    
    def _generate_text_report(self, filepath: str):
        """Generate a detailed text report."""
        with open(filepath, 'w') as f:
            f.write("LETTUCE DISEASE DATASET - EXPLORATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            if 'structure' in self.dataset_stats:
                structure = self.dataset_stats['structure']
                f.write(f"Total Images: {structure['total_images']:,}\n")
                f.write(f"Total Classes: {structure['total_classes']}\n")
                f.write(f"Active Classes: {len([c for c in structure['class_distribution'].values() if c > 0])}\n")
                f.write(f"Class Imbalance Ratio: {structure.get('imbalance_ratio', 1.0):.1f}:1\n\n")
            
            # Class Distribution
            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            if 'structure' in self.dataset_stats:
                for class_name, count in self.dataset_stats['structure']['class_distribution'].items():
                    percentage = (count / self.dataset_stats['structure']['total_images']) * 100 if self.dataset_stats['structure']['total_images'] > 0 else 0
                    f.write(f"{class_name:30} | {count:6,} images ({percentage:5.1f}%)\n")
                f.write("\n")
            
            # Hierarchical Distribution
            f.write("HIERARCHICAL DISTRIBUTION\n")
            f.write("-" * 25 + "\n")
            if 'statistics' in self.dataset_stats:
                hierarchical = self.dataset_stats['statistics']['hierarchical_distribution']
                f.write(f"Healthy:  {hierarchical['healthy_percentage']:5.1f}%\n")
                f.write(f"Disease:  {hierarchical['disease_percentage']:5.1f}%\n")
                f.write(f"Weed:     {hierarchical['weed_percentage']:5.1f}%\n\n")
            
            # Image Statistics
            f.write("IMAGE CHARACTERISTICS\n")
            f.write("-" * 20 + "\n")
            if 'images' in self.dataset_stats:
                images = self.dataset_stats['images']
                f.write(f"Total Analyzed: {images['total_analyzed']}\n")
                f.write(f"Width Range: {images['width_stats']['min']:.0f} - {images['width_stats']['max']:.0f} pixels\n")
                f.write(f"Height Range: {images['height_stats']['min']:.0f} - {images['height_stats']['max']:.0f} pixels\n")
                f.write(f"Average Aspect Ratio: {images['aspect_ratio_stats']['mean']:.2f}\n")
                f.write(f"File Size Range: {images['file_size_stats']['min']:.2f} - {images['file_size_stats']['max']:.2f} MB\n")
                f.write(f"Most Common Format: {max(images['format_distribution'], key=images['format_distribution'].get)}\n")
                f.write(f"Most Common Mode: {max(images['mode_distribution'], key=images['mode_distribution'].get)}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            self._add_recommendations(f)
    
    def _add_recommendations(self, file_handle):
        """Add analysis-based recommendations to the report."""
        recommendations = []
        
        if 'structure' in self.dataset_stats:
            imbalance_ratio = self.dataset_stats['structure'].get('imbalance_ratio', 1.0)
            
            if imbalance_ratio > 10:
                recommendations.append("⚖️ SEVERE class imbalance detected. Consider data augmentation, SMOTE, or weighted loss functions.")
            elif imbalance_ratio > 5:
                recommendations.append("⚖️ MODERATE class imbalance detected. Consider class weights in model training.")
            
            total_images = self.dataset_stats['structure']['total_images']
            if total_images < 1000:
                recommendations.append("📊 Small dataset size. Consider data augmentation and transfer learning.")
            elif total_images < 5000:
                recommendations.append("📊 Medium dataset size. Transfer learning recommended.")
        
        if 'images' in self.dataset_stats:
            width_std = self.dataset_stats['images']['width_stats']['std']
            height_std = self.dataset_stats['images']['height_stats']['std']
            
            if width_std > 200 or height_std > 200:
                recommendations.append("📏 High variability in image sizes. Consistent preprocessing crucial.")
            
            aspect_ratio_std = self.dataset_stats['images']['aspect_ratio_stats']['std']
            if aspect_ratio_std > 0.5:
                recommendations.append("📐 High variability in aspect ratios. Consider padding instead of stretching.")
        
        if not recommendations:
            recommendations.append("✅ Dataset appears well-balanced and suitable for modeling.")
        
        for i, rec in enumerate(recommendations, 1):
            file_handle.write(f"{i}. {rec}\n")
        
        file_handle.write("\n")
        file_handle.write("PREPROCESSING SUGGESTIONS\n")
        file_handle.write("-" * 25 + "\n")
        file_handle.write("1. Resize images to 224x224 for CNN compatibility\n")
        file_handle.write("2. Apply data augmentation for minority classes\n")
        file_handle.write("3. Use hierarchical approach: Crop vs Weed → Healthy vs Disease → Disease Type\n")
        file_handle.write("4. Normalize pixel values to [0, 1] range\n")
        file_handle.write("5. Consider class weights based on distribution\n")
    
    def _save_class_distribution_csv(self, filepath: str):
        """Save class distribution as CSV."""
        if 'structure' not in self.dataset_stats:
            return
        
        class_data = []
        structure = self.dataset_stats['structure']
        total_images = structure['total_images']
        
        for class_name, count in structure['class_distribution'].items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            class_type = 'Healthy' if class_name == self.config.data.healthy_class else \
                        'Weed' if class_name == self.config.data.weed_class else 'Disease'
            
            class_data.append({
                'Class': class_name,
                'Count': count,
                'Percentage': percentage,
                'Type': class_type
            })
        
        df = pd.DataFrame(class_data)
        df.to_csv(filepath, index=False)
    
    def get_preprocessing_recommendations(self) -> Dict[str, Any]:
        """Get preprocessing recommendations based on exploration results."""
        recommendations = {
            'resize_method': 'resize_and_pad',  # To preserve aspect ratio
            'target_size': (224, 224),
            'normalization': 'standard',  # [0, 1] range
            'augmentation_needed': False,
            'class_weights': None,
            'hierarchical_approach': True
        }
        
        if 'structure' in self.dataset_stats:
            imbalance_ratio = self.dataset_stats['structure'].get('imbalance_ratio', 1.0)
            
            if imbalance_ratio > 3:
                recommendations['augmentation_needed'] = True
                
                # Calculate class weights
                class_counts = self.dataset_stats['structure']['class_distribution']
                total_samples = sum(class_counts.values())
                num_classes = len([c for c in class_counts.values() if c > 0])
                
                weights = {}
                for class_name, count in class_counts.items():
                    if count > 0:
                        weights[class_name] = total_samples / (num_classes * count)
                
                recommendations['class_weights'] = weights
        
        return recommendations
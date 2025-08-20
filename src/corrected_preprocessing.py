import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class CorrectedPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def smart_resize_with_padding(self, image, target_size, pad_color=(0, 0, 0)):
        """
        Resize image while maintaining aspect ratio using padding
        This preserves disease features and leaf morphology
        """
        
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor (use minimum to ensure image fits)
        scale_w = target_width / original_width
        scale_h = target_height / original_height
        scale = min(scale_w, scale_h)  # Maintain aspect ratio
        
        # Calculate new dimensions after scaling
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image maintaining aspect ratio
        resized_image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas with target size
        canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
        
        # Calculate position to center the image
        start_x = (target_width - new_width) // 2
        start_y = (target_height - new_height) // 2
        
        # Place resized image on canvas
        canvas[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image
        
        return canvas, scale, (start_x, start_y)
    
    def intelligent_padding_color(self, image):
        """
        Calculate intelligent padding color based on image content
        Uses edge pixels to determine appropriate background
        """
        
        # Get edge pixels
        top_edge = image[0, :].mean(axis=0)
        bottom_edge = image[-1, :].mean(axis=0)
        left_edge = image[:, 0].mean(axis=0)
        right_edge = image[:, -1].mean(axis=0)
        
        # Average edge colors
        edge_color = np.mean([top_edge, bottom_edge, left_edge, right_edge], axis=0)
        
        return tuple(map(int, edge_color))
    
    def preprocess_image_correctly(self, image_path):
        """
        Correct preprocessing pipeline that preserves disease features
        """
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None, None
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get intelligent padding color
            pad_color = self.intelligent_padding_color(image)
            
            # Apply smart resize with padding
            processed_image, scale_factor, offset = self.smart_resize_with_padding(
                image, self.target_size, pad_color
            )
            
            # Normalize to [0, 1]
            normalized_image = processed_image.astype(np.float32) / 255.0
            
            # Store metadata for potential use
            metadata = {
                'original_size': image.shape[:2],
                'scale_factor': scale_factor,
                'offset': offset,
                'pad_color': pad_color
            }
            
            return normalized_image, metadata
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None
    
    def visualize_preprocessing_comparison(self, image_path):
        """
        Compare old vs new preprocessing methods
        """
        
        # Load original image
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Old method (direct resize - WRONG)
        old_resized = cv2.resize(original, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # New method (padding + resize - CORRECT)
        new_processed, metadata = self.preprocess_image_correctly(image_path)
        new_processed_uint8 = (new_processed * 255).astype(np.uint8)
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title(f'Original\n{original.shape[:2]}')
        axes[0].axis('off')
        
        axes[1].imshow(old_resized)
        axes[1].set_title(f'Direct Resize (WRONG)\n{old_resized.shape[:2]}\nDistorted Features!')
        axes[1].axis('off')
        
        axes[2].imshow(new_processed_uint8)
        axes[2].set_title(f'Padding + Resize (CORRECT)\n{new_processed_uint8.shape[:2]}\nPreserved Features!')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/preprocessing_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metadata

def test_preprocessing():
    """Test the corrected preprocessing"""
    
    # Find a sample image
    data_path = 'data'
    for class_dir in os.listdir(data_path):
        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                sample_image = os.path.join(class_path, images[0])
                break
    
    # Test preprocessing
    preprocessor = CorrectedPreprocessor()
    metadata = preprocessor.visualize_preprocessing_comparison(sample_image)
    
    print("Preprocessing Test Results:")
    print(f"Original size: {metadata['original_size']}")
    print(f"Scale factor: {metadata['scale_factor']:.3f}")
    print(f"Offset: {metadata['offset']}")
    print(f"Padding color: {metadata['pad_color']}")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    test_preprocessing()

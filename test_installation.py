#!/usr/bin/env python3
"""
Test script to verify installation.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        import skimage
        import cv2
        import PIL
        import tqdm
        import joblib
        
        print("✅ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_project_structure():
    """Test if project structure is correct."""
    required_dirs = [
        'src', 'data', 'models', 'results', 'notebooks', 'tests', 'docs', 'config',
        'src/data_processing', 'src/feature_extraction', 'src/classical_ml',
        'src/deep_learning', 'src/evaluation', 'src/utils'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Project structure is correct")
        return True

def test_pytorch():
    """Test PyTorch functionality."""
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(5, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ PyTorch with CUDA support detected")
            print(f"   GPU: {torch.cuda.get_device_name()}")
        else:
            print("✅ PyTorch CPU support detected")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Lettuce Disease Classification Installation")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure), 
        ("PyTorch Functionality", test_pytorch)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Download the lettuce disease dataset to data/raw/")
        print("2. Run: python main.py --mode explore")
        print("3. Run: python main.py --mode full")
    else:
        print("❌ Installation has issues. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Test script to verify installation and dependencies
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        import insightface
        print("✓ InsightFace imported successfully")
    except ImportError as e:
        print(f"✗ InsightFace import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    return True

def test_system_modules():
    """Test if our system modules can be imported"""
    print("\nTesting system modules...")
    
    try:
        import src.config
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Config module import failed: {e}")
        return False
    
    try:
        from src.utils import cosine_similarity, draw_bounding_box
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Utils module import failed: {e}")
        return False
    
    try:
        from src.detect_track import PersonTracker
        print("✓ Detect-track module imported successfully")
    except ImportError as e:
        print(f"✗ Detect-track module import failed: {e}")
        return False
    
    try:
        from src.face_crop import FaceProcessor
        print("✓ Face-crop module imported successfully")
    except ImportError as e:
        print(f"✗ Face-crop module import failed: {e}")
        return False
    
    try:
        from src.face_recog import FaceRecognizer
        print("✓ Face-recog module imported successfully")
    except ImportError as e:
        print(f"✗ Face-recog module import failed: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = ['models', 'assets', 'assets/faces', 'output', 'src']
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            return False
    
    return True

def test_files():
    """Test if required files exist"""
    print("\nTesting files...")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'src/config.py',
        'src/utils.py',
        'src/detect_track.py',
        'src/face_crop.py',
        'src/face_recog.py',
        'src/main.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("Real-Time Video Analytics System - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test system modules
    if not test_system_modules():
        all_passed = False
    
    # Test directories
    if not test_directories():
        all_passed = False
    
    # Test files
    if not test_files():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place your video file in assets/input_video.mp4")
        print("2. Run: python src/main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check file paths and directory structure")
        print("3. Ensure Python version is 3.8+")

if __name__ == "__main__":
    main() 
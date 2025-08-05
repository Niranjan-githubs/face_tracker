#!/usr/bin/env python3
"""
Installation test script for the Real-Time Face Tracking System
"""

import sys
import os
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('ultralytics', 'ultralytics'),
        ('deepface', 'deepface'),
        ('tensorflow', 'tensorflow'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('Pillow', 'PIL'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        ('filterpy', 'filterpy')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_gpu_support():
    """Check GPU support"""
    print("\n🖥️ Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available - {device_count} device(s)")
            print(f"   Device: {device_name}")
            return True
        else:
            print("ℹ️ CUDA not available - Using CPU")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def download_models():
    """Download required models"""
    print("\n📥 Downloading models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    try:
        # Download YOLOv8 model
        print("Downloading YOLOv8 model...")
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model downloaded")
        
        # Download DeepFace models
        print("Downloading DeepFace models...")
        from deepface import DeepFace
        import numpy as np
        # Test DeepFace functionality
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.extract_faces(img_path=test_img, detector_backend="opencv", enforce_detection=False)
        print("✅ DeepFace models downloaded")
        
        return True
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def test_imports():
    """Test importing all project modules"""
    print("\n🧪 Testing imports...")
    
    try:
        # Test core modules
        import src.config
        print("✅ config.py")
        
        import src.utils
        print("✅ utils.py")
        
        from src.detect_track import PersonTracker
        print("✅ detect_track.py")
        
        from src.face_crop import FaceProcessor
        print("✅ face_crop.py")
        
        from src.face_matcher import EnhancedFaceMatcher
        print("✅ face_matcher.py")
        
        from src.visualizer import LiveDashboard
        print("✅ visualizer.py")
        
        from src.main import FaceTrackingSystem
        print("✅ main.py")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'assets',
        'assets/faces',
        'models',
        'output'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test video capture
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access works")
            cap.release()
        else:
            print("ℹ️ Camera not available (this is normal)")
        
        # Test numpy operations
        import numpy as np
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        print("✅ NumPy operations work")
        
        # Test OpenCV operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (10, 10), (90, 90), (255, 255, 255), 2)
        print("✅ OpenCV operations work")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main installation test function"""
    print("🎯 Face Tracking System - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU Support", check_gpu_support),
        ("Directory Creation", create_directories),
        ("Model Download", download_models),
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Installation successful! System is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Place your video in assets/input_video.mp4")
        print("2. Run: python src/main.py")
        print("3. Or run demo: python demo.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Troubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Ensure internet connection for model downloads")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
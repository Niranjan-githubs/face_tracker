#!/usr/bin/env python3
"""
Demo script for the Real-Time Face Tracking & Re-Identification System
"""

import cv2
import numpy as np
import os
import sys
import time
from src.main import FaceTrackingSystem
from src.config import *

def create_sample_video():
    """Create a sample video for demonstration"""
    print("Creating sample video for demonstration...")
    
    # Create a simple video with moving objects
    width, height = 640, 480
    fps = 30
    duration = 10  # 10 seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('assets/sample_video.mp4', fourcc, fps, (width, height))
    
    # Create moving rectangles to simulate people
    for frame_num in range(fps * duration):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving rectangles (simulating people)
        t = frame_num / fps
        
        # Person 1 - moving left to right
        x1 = int(50 + t * 200)
        y1 = int(100 + 50 * np.sin(t * 2))
        cv2.rectangle(frame, (x1, y1), (x1 + 80, y1 + 120), (0, 255, 0), -1)
        
        # Person 2 - moving right to left
        x2 = int(500 - t * 150)
        y2 = int(200 + 30 * np.cos(t * 1.5))
        cv2.rectangle(frame, (x2, y2), (x2 + 80, y2 + 120), (255, 0, 0), -1)
        
        # Person 3 - moving in circle
        x3 = int(320 + 100 * np.cos(t))
        y3 = int(240 + 100 * np.sin(t))
        cv2.rectangle(frame, (x3, y3), (x3 + 80, y3 + 120), (0, 0, 255), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Sample Video for Face Tracking Demo", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print("Sample video created: assets/sample_video.mp4")

def run_demo():
    """Run the face tracking demo"""
    print("🎯 Real-Time Face Tracking & Re-Identification Demo")
    print("=" * 60)
    
    # Check if sample video exists, create if not
    if not os.path.exists('assets/sample_video.mp4'):
        create_sample_video()
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    # Copy sample video to input location
    if os.path.exists('assets/sample_video.mp4'):
        import shutil
        shutil.copy('assets/sample_video.mp4', INPUT_VIDEO)
        print(f"Using sample video: {INPUT_VIDEO}")
    else:
        print("❌ No sample video found. Please place a video file in assets/input_video.mp4")
        return
    
    # Initialize system
    print("\n🚀 Initializing Face Tracking System...")
    system = FaceTrackingSystem()
    
    # Run the system
    print("\n📊 Starting real-time processing...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'c' to clear face cache")
    print("\n" + "=" * 60)
    
    try:
        system.run(INPUT_VIDEO)
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        return
    
    print("\n✅ Demo completed successfully!")
    print(f"📁 Output video saved to: {OUTPUT_VIDEO}")

def show_system_info():
    """Display system information and capabilities"""
    print("🔧 System Information")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Check CUDA availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Check DeepFace
    try:
        import deepface
        print(f"DeepFace version: {deepface.__version__}")
    except ImportError:
        print("DeepFace: Not installed")
    
    print("\n🎯 Key Features:")
    print("  ✅ Real-time person detection (YOLOv8)")
    print("  ✅ Multi-object tracking (ByteTracker)")
    print("  ✅ Face detection and cropping")
    print("  ✅ Persistent face re-identification")
    print("  ✅ Live dashboard with statistics")
    print("  ✅ Video export with overlays")
    print("  ✅ Reappearance detection")
    print("  ✅ Performance monitoring")

def main():
    """Main demo function"""
    print("🎯 Real-Time Face Tracking & Re-Identification System")
    print("🏆 Built for Hackathon Excellence")
    print("=" * 60)
    
    # Show system info
    show_system_info()
    
    print("\n" + "=" * 60)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Run the demo with sample video")
    print("2. Run with your own video (place in assets/input_video.mp4)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        if os.path.exists(INPUT_VIDEO):
            print(f"\n🚀 Running with your video: {INPUT_VIDEO}")
            system = FaceTrackingSystem()
            system.run(INPUT_VIDEO)
        else:
            print(f"\n❌ Video not found at {INPUT_VIDEO}")
            print("Please place your video file in assets/input_video.mp4")
    elif choice == "3":
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice. Running demo...")
        run_demo()

if __name__ == "__main__":
    main() 
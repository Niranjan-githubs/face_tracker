#!/usr/bin/env python3
"""
Test script to generate proper MP4 output with face tracking
"""

import sys
import os
import signal
import time
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import FaceTrackingSystem
from config import INPUT_VIDEO

def signal_handler(signum, frame):
    print("\n🛑 Interrupted by user")
    sys.exit(0)

def test_mp4_output():
    """Test MP4 output generation"""
    print("🎬 MP4 Output Test")
    print("=" * 50)
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize system
        system = FaceTrackingSystem()
        
        # Override the run method to process more frames
        def mp4_test_run(video_path):
            """Run with enough frames for MP4 output"""
            try:
                system.initialize_video(video_path)
                
                print("Starting MP4 output test...")
                print("Processing 20 frames for proper MP4 output...")
                
                frame_count = 0
                max_frames = 20
                
                while frame_count < max_frames:
                    ret, frame = system.cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame = system.process_frame(frame)
                    
                    # Store processed frame for later video writing
                    system.processed_frames.append(processed_frame.copy())
                    
                    # Display dashboard
                    system.dashboard.show_dashboard(processed_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    
                    frame_count += 1
                    print(f"Processed frame {frame_count}/{max_frames}")
                    
                    # Small delay to make it visible
                    time.sleep(0.05)
                
                print(f"✅ Processed {frame_count} frames successfully!")
                
            finally:
                system.cleanup()
        
        # Replace the run method
        system.run = mp4_test_run
        
        # Run the test
        system.run(INPUT_VIDEO)
        
        print("\n🎉 MP4 output test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mp4_output() 
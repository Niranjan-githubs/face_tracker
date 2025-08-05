#!/usr/bin/env python3
"""
Test script to demonstrate enhanced person ID display and reduced face flickering
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

def test_enhanced_display():
    """Test enhanced person ID display and face tracking"""
    print("🎯 Enhanced Display Test")
    print("=" * 50)
    print("✅ Person IDs now displayed prominently")
    print("✅ Face flickering reduced with persistence")
    print("✅ Face boxes show person association")
    print("=" * 50)
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize system
        system = FaceTrackingSystem()
        
        # Override the run method to process more frames for better demonstration
        def enhanced_run(self, video_path: str):
            """Enhanced run method for better display testing"""
            self.initialize_video(video_path)
            
            print("🎬 Processing video with enhanced display...")
            print("📊 Person IDs: Prominent green boxes with white text")
            print("👤 Face IDs: Blue boxes with person association")
            print("🔄 Face persistence: Reduced flickering")
            
            frame_count = 0
            max_frames = 30  # Process more frames for better demonstration
            
            while True:
                ret, frame = self.cap.read()
                if not ret or frame_count >= max_frames:
                    break
                
                frame_count += 1
                print(f"Processing frame {frame_count}/{max_frames}")
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Store for video output
                self.processed_frames.append(processed_frame.copy())
                
                # Display frame
                cv2.imshow('Enhanced Face Tracking', processed_frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Cleanup
            self.cleanup()
            cv2.destroyAllWindows()
        
        # Replace the run method temporarily
        original_run = system.run
        system.run = enhanced_run.__get__(system, FaceTrackingSystem)
        
        # Run the enhanced test
        system.run(INPUT_VIDEO)
        
        # Restore original method
        system.run = original_run
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_display() 
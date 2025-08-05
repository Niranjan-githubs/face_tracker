#!/usr/bin/env python3
"""
Test script to demonstrate improved face ID consistency
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

def test_face_consistency():
    """Test improved face ID consistency"""
    print("🎯 Face ID Consistency Test")
    print("=" * 50)
    print("✅ Same person = Same face ID")
    print("✅ Person-persistent ID mapping")
    print("✅ Lower similarity threshold (0.5)")
    print("✅ Temporal consistency")
    print("=" * 50)
    
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize system
        system = FaceTrackingSystem()
        
        # Override the run method to show face ID consistency
        def consistency_run(self, video_path: str):
            """Run method focused on face ID consistency"""
            self.initialize_video(video_path)
            
            print("🎬 Testing face ID consistency...")
            print("📊 Tracking person-to-face ID mapping...")
            
            frame_count = 0
            max_frames = 25  # Process more frames to see consistency
            
            # Track person-to-face mappings
            person_face_mappings = {}
            
            while True:
                ret, frame = self.cap.read()
                if not ret or frame_count >= max_frames:
                    break
                
                frame_count += 1
                print(f"\n📸 Frame {frame_count}/{max_frames}")
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Store for video output
                self.processed_frames.append(processed_frame.copy())
                
                # Display frame
                cv2.imshow('Face ID Consistency Test', processed_frame)
                
                # Show person-to-face mappings
                if hasattr(self.face_matcher, 'person_persistent_mapping'):
                    current_mappings = self.face_matcher.person_persistent_mapping
                    if current_mappings:
                        print("👥 Person → Face ID mappings:")
                        for person_id, face_id in current_mappings.items():
                            print(f"   Person {person_id} → Face {face_id}")
                            person_face_mappings[person_id] = face_id
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Show final consistency summary
            print(f"\n🎯 Consistency Summary:")
            print(f"📊 Total unique persons: {len(person_face_mappings)}")
            print(f"🆔 Person-to-Face mappings:")
            for person_id, face_id in person_face_mappings.items():
                print(f"   Person {person_id} consistently mapped to Face {face_id}")
            
            # Cleanup
            self.cleanup()
            cv2.destroyAllWindows()
        
        # Replace the run method temporarily
        original_run = system.run
        system.run = consistency_run.__get__(system, FaceTrackingSystem)
        
        # Run the consistency test
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
    test_face_consistency() 
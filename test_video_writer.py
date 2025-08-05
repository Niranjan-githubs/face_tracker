#!/usr/bin/env python3
"""
Simple test script to verify video writer functionality
"""

import cv2
import numpy as np
import os

def test_video_writer():
    """Test video writer with the new naming convention"""
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 3  # 3 seconds
    
    # Extract input video name (simulating the main script)
    input_video_name = "input_video"  # This would come from the actual input file
    
    # Create output path with new naming convention
    output_path = f"outputs/{input_video_name}_bounded.avi"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not initialize video writer for {output_path}")
        return False
    
    print(f"Video writer initialized: {output_path}")
    
    # Create test frames
    for frame_num in range(fps * duration):
        # Create a simple test frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some moving elements
        t = frame_num / fps
        
        # Moving rectangle
        x = int(50 + t * 200)
        y = int(100 + 50 * np.sin(t * 2))
        cv2.rectangle(frame, (x, y), (x + 80, y + 120), (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        video_writer.write(frame)
        
        if frame_num % 30 == 0:
            print(f"Written frame {frame_num}")
    
    # Release video writer
    video_writer.release()
    
    # Check file size
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Video saved: {output_path}")
        print(f"File size: {file_size} bytes")
        return file_size > 1000  # Should be much larger than 1000 bytes
    else:
        print(f"Error: Video file not created")
        return False

if __name__ == "__main__":
    success = test_video_writer()
    if success:
        print("✅ Video writer test PASSED")
    else:
        print("❌ Video writer test FAILED") 
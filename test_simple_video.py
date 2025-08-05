#!/usr/bin/env python3
"""
Simple test to verify video writing functionality without heavy model loading
"""

import cv2
import numpy as np
import os
import time
import subprocess

def create_test_frames(width=1920, height=1080, num_frames=50):
    """Create test frames with moving objects to simulate face tracking output"""
    frames = []
    
    for i in range(num_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving rectangles to simulate people and faces
        t = i / 30.0  # Time progression
        
        # Person 1 - moving left to right
        x1 = int(100 + t * 300)
        y1 = int(200 + 50 * np.sin(t * 2))
        cv2.rectangle(frame, (x1, y1), (x1 + 120, y1 + 200), (0, 255, 0), 3)
        cv2.putText(frame, f"Person 1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Face 1 - moving with person 1
        face_x1 = x1 + 30
        face_y1 = y1 + 20
        cv2.rectangle(frame, (face_x1, face_y1), (face_x1 + 60, face_y1 + 60), (255, 0, 0), 2)
        cv2.putText(frame, f"Face 1", (face_x1, face_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Person 2 - moving right to left
        x2 = int(800 - t * 200)
        y2 = int(400 + 30 * np.cos(t * 1.5))
        cv2.rectangle(frame, (x2, y2), (x2 + 120, y2 + 200), (0, 255, 0), 3)
        cv2.putText(frame, f"Person 2", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Face 2 - moving with person 2
        face_x2 = x2 + 30
        face_y2 = y2 + 20
        cv2.rectangle(frame, (face_x2, face_y2), (face_x2 + 60, face_y2 + 60), (255, 0, 0), 2)
        cv2.putText(frame, f"Face 2", (face_x2, face_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add statistics overlay
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"People: 2", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: 2", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: 30.0", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames

def write_video_to_mp4(frames, fps=30, input_video_name="input_video"):
    """Write frames to MP4 using the same method as the main system"""
    if not frames:
        print("No frames to write")
        return False
    
    ensure_directory("outputs")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create output path
    output_path = f"outputs/{input_video_name}_bounded.mp4"
    
    print(f"Writing {len(frames)} frames to {output_path}")
    print(f"Frame size: {width}x{height}, FPS: {fps}")
    
    try:
        # First write to temporary AVI file (OpenCV is more reliable with AVI)
        temp_avi_path = f"outputs/{input_video_name}_temp.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_writer = cv2.VideoWriter(
            temp_avi_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        if not temp_writer.isOpened():
            raise ValueError(f"Could not initialize temporary video writer")
        
        # Write all frames
        for i, frame in enumerate(frames):
            temp_writer.write(frame)
            if i % 10 == 0:
                print(f"Written frame {i}/{len(frames)}")
        
        temp_writer.release()
        
        # Convert AVI to MP4 using ffmpeg
        cmd = [
            'ffmpeg', '-i', temp_avi_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            output_path
        ]
        
        print(f"Converting to MP4...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Remove temporary AVI file
        if os.path.exists(temp_avi_path):
            os.remove(temp_avi_path)
        
        file_size = os.path.getsize(output_path)
        print(f"✅ Video saved successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"🎬 Frames written: {len(frames)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing video: {e}")
        return False

def ensure_directory(directory):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    print("🎬 Simple Video Writing Test")
    print("=" * 50)
    
    # Create test frames
    print("Creating test frames...")
    frames = create_test_frames(width=1920, height=1080, num_frames=100)
    print(f"Created {len(frames)} test frames")
    
    # Write video
    success = write_video_to_mp4(frames, fps=30, input_video_name="input_video")
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("The video writing functionality is working correctly.")
    else:
        print("\n❌ Test failed!")
        print("There's an issue with the video writing functionality.") 
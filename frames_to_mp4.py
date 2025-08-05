#!/usr/bin/env python3
"""
Convert frames to MP4 using image sequence method (more reliable on macOS)
"""

import os
import subprocess
import sys
import cv2
import numpy as np

def frames_to_mp4(frames, output_path, fps=30):
    """Convert frames to MP4 using image sequence method"""
    
    if not frames:
        print("No frames to process")
        return False
    
    # Create temporary directory for frames
    temp_dir = "outputs/temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Saving {len(frames)} frames as images...")
    
    # Save frames as images
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
    
    print(f"Saved {len(frames)} frames to {temp_dir}")
    
    try:
        # Convert image sequence to MP4 using ffmpeg
        input_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        print(f"Converting to MP4: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Clean up temporary files
        for i in range(len(frames)):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        # Remove temp directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            file_size = os.path.getsize(output_path)
            print(f"✅ MP4 created successfully!")
            print(f"📁 Output: {output_path}")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            return True
        else:
            print("❌ MP4 file was not created or is too small")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        
        # Clean up on error
        for i in range(len(frames)):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Frames to MP4 Converter")
    print("=" * 40)
    
    # Test with some sample frames
    frames = []
    for i in range(20):
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)
    
    output_path = "outputs/test_output.mp4"
    success = frames_to_mp4(frames, output_path, fps=30)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1) 
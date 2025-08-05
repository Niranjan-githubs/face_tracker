#!/usr/bin/env python3
"""
Convert AVI to MP4 using a reliable method
"""

import os
import subprocess
import sys

def convert_avi_to_mp4():
    """Convert the working AVI file to MP4"""
    avi_file = "outputs/input_video_bounded.avi"
    mp4_file = "outputs/input_video_bounded.mp4"
    
    if not os.path.exists(avi_file):
        print(f"❌ AVI file not found: {avi_file}")
        return False
    
    file_size = os.path.getsize(avi_file)
    print(f"📁 Found AVI file: {avi_file}")
    print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    if file_size < 1000:
        print("❌ AVI file is too small, likely corrupted")
        return False
    
    print(f"🔄 Converting {avi_file} to {mp4_file}...")
    
    try:
        # Use ffmpeg with more robust settings
        cmd = [
            'ffmpeg',
            '-i', avi_file,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',  # Ensure compatibility
            '-y',  # Overwrite output file
            mp4_file
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if os.path.exists(mp4_file):
            mp4_size = os.path.getsize(mp4_file)
            print(f"✅ Conversion successful!")
            print(f"📁 MP4 file: {mp4_file}")
            print(f"📊 MP4 size: {mp4_size:,} bytes ({mp4_size/1024/1024:.1f} MB)")
            return True
        else:
            print("❌ MP4 file was not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🎬 AVI to MP4 Converter")
    print("=" * 40)
    
    success = convert_avi_to_mp4()
    
    if success:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1) 
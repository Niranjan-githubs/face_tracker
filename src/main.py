"""
Main video processing pipeline for real-time face tracking and re-identification
"""

import cv2
import numpy as np
import time
from typing import Dict, List
import os
import sys
from src.config import *
from src.detect_track import PersonTracker
from src.face_crop import FaceProcessor
from src.face_matcher import EnhancedFaceMatcher
from src.visualizer import LiveDashboard
from src.utils import *

class FaceTrackingSystem:
    """Complete real-time face tracking and re-identification system"""
    
    def __init__(self):
        """Initialize the complete face tracking system"""
        self.person_tracker = PersonTracker()
        self.face_processor = FaceProcessor()
        self.face_matcher = EnhancedFaceMatcher()
        self.dashboard = LiveDashboard()
        
        # Video processing
        self.cap = None
        self.video_writer = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Store processed frames for later video writing
        self.processed_frames = []
        self.video_properties = None
        
        # Performance monitoring
        self.fps_counter = FPS()
        
        # Statistics
        self.stats = {
            'FPS': 0,
            'Total Frames': 0,
            'Total People': 0,
            'Active IDs': 0,
            'Faces Detected': 0,
            'Unique Faces': 0,
            'Matches': 0,
            'Processing Time': 0
        }
        
    def initialize_video(self, video_path: str):
        """Initialize video capture and writer"""
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract input video name for output naming
        input_video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Store video properties for later video writing
        self.video_properties = {
            'width': width,
            'height': height,
            'fps': fps,
            'input_video_name': input_video_name
        }
        
        print(f"Video properties stored: {width}x{height} @ {fps} FPS")
        
        # Initialize analytics tracking
        self.analytics = {
            'video_info': {
                'input_path': video_path,
                'output_path': f"outputs/{input_video_name}_bounded.mp4",  # Will be MP4
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'processing_start_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'tracking_data': [],
            'face_analytics': {
                'total_faces_detected': 0,
                'unique_faces_tracked': 0,
                'total_reidentifications': 0,
                'face_detection_frames': 0,
                'average_faces_per_frame': 0
            },
            'performance_metrics': {
                'total_processing_time': 0,
                'average_fps': 0,
                'frames_processed': 0
            }
        }
        
        print(f"Video initialized: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Output: outputs/{input_video_name}_bounded.mp4")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the complete pipeline"""
        start_time = time.time()
        
        # Step 1: Detect and track people
        tracked_persons = self.person_tracker.process_frame(frame)
        
        # Step 2: Detect faces in tracked persons
        detected_faces = self.face_processor.process_persons(frame, tracked_persons)
        
        # Step 3: Match faces with persistent IDs
        processed_faces = self.face_matcher.process_frame_faces(detected_faces)
        
        # Step 4: Create dashboard
        dashboard = self.dashboard.create_dashboard(frame, tracked_persons, processed_faces, self.stats)
        
        # Step 5: Update statistics
        self.update_stats(tracked_persons, processed_faces, time.time() - start_time)
        
        # Step 6: Update analytics
        self.update_analytics(tracked_persons, processed_faces, time.time() - start_time)
        
        return dashboard
    
    def draw_results(self, frame: np.ndarray, 
                    tracked_persons: List, 
                    processed_faces: List) -> np.ndarray:
        """Draw bounding boxes, IDs, and face panels"""
        # Draw person bounding boxes
        if DRAW_BOUNDING_BOXES:
            for bbox, person_id, confidence in tracked_persons:
                frame = draw_bounding_box(frame, bbox, f"Person {person_id}", 
                                        PERSON_BOX_COLOR, confidence)
        
        # Draw face bounding boxes and prepare face panel
        face_panel_faces = []
        for face_img, face_bbox, person_id, face_id, confidence, person_conf, face_conf in processed_faces:
            if DRAW_BOUNDING_BOXES:
                frame = draw_bounding_box(frame, face_bbox, f"Face {face_id}", 
                                        FACE_BOX_COLOR, confidence)
            
            face_panel_faces.append((face_img, face_id, confidence))
        
        # Create and display face panel
        if SHOW_FACE_PANELS and face_panel_faces:
            self.face_panel = create_face_panel(face_panel_faces)
            if self.face_panel is not None:
                # Display panel in separate window
                cv2.imshow('Face Panel', self.face_panel)
        
        # Draw statistics
        frame = draw_stats(frame, self.stats)
        
        return frame
    
    def update_stats(self, tracked_persons: List, processed_faces: List, processing_time: float):
        """Update processing statistics"""
        self.stats['Total People'] = len(tracked_persons)
        self.stats['Faces Detected'] = len(processed_faces)
        self.stats['Total Frames'] = self.frame_count
        self.stats['Processing Time'] = processing_time * 1000  # Convert to milliseconds
        
        # Get face matcher stats
        face_stats = self.face_matcher.get_statistics()
        self.stats['Active IDs'] = face_stats['active_persistent_faces']
        self.stats['Unique Faces'] = face_stats['total_unique_faces']
        self.stats['Matches'] = face_stats['total_matches']
        
        # Update FPS
        self.stats['FPS'] = self.fps_counter.update()
    
    def update_analytics(self, tracked_persons: List, processed_faces: List, processing_time: float):
        """Update comprehensive analytics data"""
        # Update performance metrics
        self.analytics['performance_metrics']['total_processing_time'] += processing_time
        self.analytics['performance_metrics']['frames_processed'] += 1
        
        # Update face analytics
        if processed_faces:
            self.analytics['face_analytics']['face_detection_frames'] += 1
            self.analytics['face_analytics']['total_faces_detected'] += len(processed_faces)
        
        # Get face matcher statistics
        face_stats = self.face_matcher.get_statistics()
        self.analytics['face_analytics']['unique_faces_tracked'] = face_stats['total_unique_faces']
        self.analytics['face_analytics']['total_reidentifications'] = face_stats['total_matches']
        
        # Store frame-level tracking data
        frame_data = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'processing_time': processing_time,
            'persons_detected': len(tracked_persons),
            'faces_detected': len(processed_faces),
            'active_persistent_faces': face_stats['active_persistent_faces'],
            'person_tracks': [
                {
                    'person_id': person_id,
                    'bbox': bbox,
                    'confidence': conf
                } for bbox, person_id, conf in tracked_persons
            ],
            'face_tracks': [
                {
                    'face_id': persistent_id,
                    'person_id': person_id,
                    'bbox': face_bbox,
                    'confidence': confidence,
                    'embedding_similarity': confidence
                } for face_img, face_bbox, person_id, persistent_id, confidence, person_conf, face_conf in processed_faces
            ]
        }
        self.analytics['tracking_data'].append(frame_data)
    
    def save_face_images(self, processed_faces: List):
        """Save individual face images"""
        for face_img, face_bbox, person_id, face_id, confidence, person_conf, face_conf in processed_faces:
            save_face_image(face_img, face_id, self.frame_count)
    
    def save_analytics_report(self):
        """Save comprehensive analytics report"""
        # Calculate final metrics
        total_frames = self.analytics['performance_metrics']['frames_processed']
        if total_frames > 0:
            self.analytics['performance_metrics']['average_fps'] = total_frames / self.analytics['performance_metrics']['total_processing_time']
            self.analytics['face_analytics']['average_faces_per_frame'] = self.analytics['face_analytics']['total_faces_detected'] / total_frames
        
        # Add processing end time
        self.analytics['video_info']['processing_end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.analytics['video_info']['total_processing_duration'] = self.analytics['performance_metrics']['total_processing_time']
        
        # Save JSON report
        import json
        analytics_path = "outputs/face_tracking_analytics.json"
        with open(analytics_path, 'w') as f:
            json.dump(self.analytics, f, indent=2, default=str)
        
        # Save summary report
        summary_path = "outputs/face_tracking_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("FACE TRACKING ANALYTICS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video Information:\n")
            f.write(f"  Input: {self.analytics['video_info']['input_path']}\n")
            f.write(f"  Output: {self.analytics['video_info']['output_path']}\n")
            f.write(f"  Resolution: {self.analytics['video_info']['width']}x{self.analytics['video_info']['height']}\n")
            f.write(f"  FPS: {self.analytics['video_info']['fps']}\n")
            f.write(f"  Total Frames: {self.analytics['video_info']['total_frames']}\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  Frames Processed: {self.analytics['performance_metrics']['frames_processed']}\n")
            f.write(f"  Average FPS: {self.analytics['performance_metrics']['average_fps']:.2f}\n")
            f.write(f"  Total Processing Time: {self.analytics['performance_metrics']['total_processing_time']:.2f}s\n\n")
            
            f.write(f"Face Analytics:\n")
            f.write(f"  Total Faces Detected: {self.analytics['face_analytics']['total_faces_detected']}\n")
            f.write(f"  Unique Faces Tracked: {self.analytics['face_analytics']['unique_faces_tracked']}\n")
            f.write(f"  Total Re-identifications: {self.analytics['face_analytics']['total_reidentifications']}\n")
            f.write(f"  Face Detection Frames: {self.analytics['face_analytics']['face_detection_frames']}\n")
            f.write(f"  Average Faces per Frame: {self.analytics['face_analytics']['average_faces_per_frame']:.2f}\n\n")
            
            f.write(f"Processing Details:\n")
            f.write(f"  Start Time: {self.analytics['video_info']['processing_start_time']}\n")
            f.write(f"  End Time: {self.analytics['video_info']['processing_end_time']}\n")
            f.write(f"  Duration: {self.analytics['video_info']['total_processing_duration']:.2f}s\n")
        
        print(f"Analytics saved to: {analytics_path}")
        print(f"Summary saved to: {summary_path}")
    
    def write_video_output(self):
        """Write all processed frames to MP4 video"""
        if not self.processed_frames or not self.video_properties:
            print("No frames to write or video properties missing")
            return
        
        # Check if we have enough frames for a proper video
        if len(self.processed_frames) < 5:
            print(f"⚠️  Only {len(self.processed_frames)} frames processed. Need at least 5 frames for proper video.")
            print("Saving as AVI format instead of MP4.")
            self._save_as_avi()
            return
        
        ensure_directory("outputs")
        
        # Create output path
        input_video_name = self.video_properties['input_video_name']
        output_path = f"outputs/{input_video_name}_bounded.mp4"
        
        print(f"Writing {len(self.processed_frames)} frames to {output_path}")
        
        try:
            # Try direct MP4 writing with different codecs
            success = False
            
            # Try different codecs - start with more reliable ones
            codecs_to_try = [
                ('XVID', 'XVID'),  # XVID - very reliable
                ('MJPG', 'MJPG'),  # Motion JPEG - more reliable
                ('mp4v', 'mp4v'),  # MP4V - less reliable
                ('avc1', 'avc1')   # AVC1 - less reliable
            ]
            
            for codec_name, fourcc_code in codecs_to_try:
                try:
                    print(f"Trying codec: {codec_name}")
                    
                    # Use different file extensions based on codec
                    if codec_name in ['mp4v', 'avc1']:
                        temp_output_path = output_path
                    elif codec_name == 'XVID':
                        # XVID needs .avi extension
                        temp_output_path = f"outputs/{input_video_name}_temp.avi"
                    else:
                        # For other codecs, use temporary file
                        temp_output_path = f"outputs/{input_video_name}_temp.{codec_name.lower()}"
                    
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                    writer = cv2.VideoWriter(
                        temp_output_path, 
                        fourcc, 
                        self.video_properties['fps'], 
                        (self.video_properties['width'], self.video_properties['height'])
                    )
                    
                    if writer.isOpened():
                        # Write all frames
                        for frame in self.processed_frames:
                            writer.write(frame)
                        
                        writer.release()
                        
                        # Check if file was created and has content
                        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 1000:
                            # If it's not already an MP4, convert it
                            if temp_output_path != output_path:
                                try:
                                    import subprocess
                                    cmd = [
                                        'ffmpeg', '-i', temp_output_path,
                                        '-c:v', 'libx264',
                                        '-preset', 'ultrafast',
                                        '-crf', '23',
                                        '-y',
                                        output_path
                                    ]
                                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                                    
                                    # Remove temporary file
                                    if os.path.exists(temp_output_path):
                                        os.remove(temp_output_path)
                                    
                                    success = True
                                    print(f"✅ Successfully converted to MP4 with codec: {codec_name}")
                                    break
                                except Exception as conv_e:
                                    print(f"❌ Conversion failed for {codec_name}: {conv_e}")
                                    if os.path.exists(temp_output_path):
                                        os.remove(temp_output_path)
                                    continue
                            else:
                                success = True
                                print(f"✅ Successfully wrote MP4 with codec: {codec_name}")
                                break
                        else:
                            print(f"❌ Codec {codec_name} failed - file too small or empty")
                            if os.path.exists(temp_output_path):
                                os.remove(temp_output_path)
                    else:
                        print(f"❌ Could not open writer with codec: {codec_name}")
                        
                except Exception as e:
                    print(f"❌ Codec {codec_name} failed: {e}")
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                    continue
            
            if success:
                # Update analytics
                self.analytics['video_info']['output_path'] = output_path
                
                file_size = os.path.getsize(output_path)
                print(f"✅ MP4 video saved successfully!")
                print(f"📁 Output: {output_path}")
                print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                print(f"🎬 Frames written: {len(self.processed_frames)}")
            else:
                print("❌ All MP4 codecs failed, falling back to AVI")
                self._save_as_avi()
            
        except Exception as e:
            print(f"❌ Error writing video: {e}")
            # Fallback: save as AVI if MP4 conversion fails
            self._save_as_avi()
    
    def _save_as_avi(self):
        """Save video as AVI format"""
        input_video_name = self.video_properties['input_video_name']
        fallback_path = f"outputs/{input_video_name}_bounded.avi"
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                fallback_path, 
                fourcc, 
                self.video_properties['fps'], 
                (self.video_properties['width'], self.video_properties['height'])
            )
            
            if not writer.isOpened():
                raise ValueError(f"Could not initialize AVI video writer")
            
            # Write all frames
            for frame in self.processed_frames:
                writer.write(frame)
            
            writer.release()
            
            # Update analytics
            self.analytics['video_info']['output_path'] = fallback_path
            
            file_size = os.path.getsize(fallback_path)
            print(f"✅ AVI video saved successfully!")
            print(f"📁 Output: {fallback_path}")
            print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print(f"🎬 Frames written: {len(self.processed_frames)}")
            
        except Exception as e:
            print(f"❌ Error saving AVI video: {e}")
    
    def run(self, video_path: str):
        """Run the complete face tracking system"""
        try:
            self.initialize_video(video_path)
            
            print("Starting face tracking system...")
            print("Press 'q' to quit, 's' to save current frame, 'c' to clear face cache")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Store processed frame for later video writing
                self.processed_frames.append(processed_frame.copy())
                
                # Display dashboard
                self.dashboard.show_dashboard(processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    ensure_directory("output")
                    cv2.imwrite(f"output/frame_{self.frame_count}.jpg", processed_frame)
                    print(f"Saved frame {self.frame_count}")
                elif key == ord('c'):
                    # Clear face cache
                    self.face_matcher.clear_all_data()
                    print("Face cache cleared")
                
                self.frame_count += 1
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames, FPS: {self.stats['FPS']:.1f}")
                    print(f"People: {self.stats['Total People']}, Faces: {self.stats['Faces Detected']}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and save analytics"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Write video output
        self.write_video_output()
        
        # Save analytics report
        self.save_analytics_report()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {self.stats['FPS']:.1f}")
        print(f"Output video saved to: {self.analytics['video_info']['output_path']}")
        print(f"Analytics saved to: outputs/face_tracking_analytics.json")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        # Print analytics summary
        print(f"\nAnalytics Summary:")
        print(f"  Total Faces Detected: {self.analytics['face_analytics']['total_faces_detected']}")
        print(f"  Unique Faces Tracked: {self.analytics['face_analytics']['unique_faces_tracked']}")
        print(f"  Total Re-identifications: {self.analytics['face_analytics']['total_reidentifications']}")
        print(f"  Average Faces per Frame: {self.analytics['face_analytics']['average_faces_per_frame']:.2f}")

def main():
    """Main function"""
    # Create output directories
    ensure_directory("output")
    ensure_directory("assets")
    ensure_directory("assets/faces")
    ensure_directory("models")
    
    # Check if input video exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video not found at {INPUT_VIDEO}")
        print("Please place your video file in the assets directory as 'input_video.mp4'")
        return
    
    # Initialize and run system
    system = FaceTrackingSystem()
    system.run(INPUT_VIDEO)

if __name__ == "__main__":
    main() 
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
from src.face_recog import FaceRecognizer
from src.utils import *

class FaceTrackingSystem:
    """Complete real-time face tracking and re-identification system"""
    
    def __init__(self):
        """Initialize the complete face tracking system"""
        self.person_tracker = PersonTracker()
        self.face_processor = FaceProcessor()
        self.face_recognizer = FaceRecognizer()
        
        # Video processing
        self.cap = None
        self.video_writer = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance monitoring
        self.fps_counter = FPS()
        
        # Statistics
        self.stats = {
            'FPS': 0,
            'Total Frames': 0,
            'People Detected': 0,
            'Faces Detected': 0,
            'Unique Faces': 0,
            'Embedding Matches': 0
        }
        
        # Face panel
        self.face_panel = None
        
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
        
        # Initialize video writer
        ensure_directory(os.path.dirname(OUTPUT_VIDEO))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        
        print(f"Video initialized: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Output: {OUTPUT_VIDEO}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the complete pipeline"""
        # Step 1: Detect and track people
        tracked_persons = self.person_tracker.process_frame(frame)
        
        # Step 2: Detect faces in tracked persons
        detected_faces = self.face_processor.process_persons(frame, tracked_persons)
        
        # Step 3: Recognize and re-identify faces
        processed_faces = self.face_recognizer.process_faces(detected_faces)
        
        # Step 4: Draw results
        frame = self.draw_results(frame, tracked_persons, processed_faces)
        
        # Step 5: Update statistics
        self.update_stats(tracked_persons, processed_faces)
        
        return frame
    
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
    
    def update_stats(self, tracked_persons: List, processed_faces: List):
        """Update processing statistics"""
        self.stats['People Detected'] = len(tracked_persons)
        self.stats['Faces Detected'] = len(processed_faces)
        self.stats['Total Frames'] = self.frame_count
        
        # Get face recognition stats
        face_stats = self.face_recognizer.get_face_stats()
        self.stats['Unique Faces'] = face_stats['unique_faces']
        self.stats['Embedding Matches'] = face_stats['embedding_matches']
        
        # Update FPS
        self.stats['FPS'] = self.fps_counter.update()
    
    def save_face_images(self, processed_faces: List):
        """Save individual face images"""
        for face_img, face_bbox, person_id, face_id, confidence, person_conf, face_conf in processed_faces:
            save_face_image(face_img, face_id, self.frame_count)
    
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
                
                # Write frame
                self.video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Face Tracking', processed_frame)
                
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
                    self.face_recognizer.clear_cache()
                    print("Face cache cleared")
                
                self.frame_count += 1
                
                # Print progress every 100 frames
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames, FPS: {self.stats['FPS']:.1f}")
                    print(f"People: {self.stats['People Detected']}, Faces: {self.stats['Faces Detected']}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {self.stats['FPS']:.1f}")
        print(f"Output video saved to: {OUTPUT_VIDEO}")
        
        # Print final statistics
        print(f"\nFinal Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")

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
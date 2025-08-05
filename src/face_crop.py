"""
Face detection and cropping within person bounding boxes using DeepFace
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from src.config import *

class FaceDetector:
    """YOLOv8-face detector"""
    def __init__(self):
        """Initialize YOLOv8-face detector"""
        # Download yolov8n-face.pt from a YOLOv8-face repo and place in models/
        self.model = YOLO('models/yolov8n-face.pt')

    def detect_faces(self, frame: np.ndarray, person_bbox: List[int] = None) -> List[Dict]:
        """Detect faces in the frame or within a person bounding box (if provided)"""
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            region = frame[y1:y2, x1:x2]
            region_offset = (x1, y1)
        else:
            region = frame
            region_offset = (0, 0)

        results = self.model(region)
        faces = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            # Adjust coordinates if region is a crop
            x1_full = x1 + region_offset[0]
            y1_full = y1 + region_offset[1]
            x2_full = x2 + region_offset[0]
            y2_full = y2 + region_offset[1]
            face_img = frame[y1_full:y2_full, x1_full:x2_full]
            # Check minimum face size
            if (x2_full - x1_full) >= MIN_FACE_SIZE and (y2_full - y1_full) >= MIN_FACE_SIZE:
                if face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    # For compatibility with rest of pipeline
                    class SimpleFace:
                        def __init__(self, bbox, confidence):
                            self.bbox = np.array(bbox)
                            self.det_score = float(box[4]) if len(box) > 4 else 0.9
                    face_obj = SimpleFace([x1_full, y1_full, x2_full, y2_full], float(box[4]) if len(box) > 4 else 0.9)
                    faces.append({
                        'bbox': [x1_full, y1_full, x2_full, y2_full],
                        'face_img': face_img,
                        'face_obj': face_obj,
                        'confidence': float(box[4]) if len(box) > 4 else 0.9
                    })
        return faces

class FaceProcessor:
    """Complete face processing system"""
    def __init__(self):
        """Initialize face processing system"""
        self.detector = FaceDetector()
        self.frame_count = 0
        
    def process_persons(self, frame: np.ndarray, 
                       tracked_persons: List[Tuple]) -> List[Tuple]:
        """Process all tracked persons for face detection"""
        self.frame_count += 1
        
        # Only detect faces every N frames for performance
        if self.frame_count % FACE_DETECTION_INTERVAL != 0:
            return []
        
        all_faces = []
        
        if tracked_persons:
            for person_bbox, person_id, person_conf in tracked_persons:
                faces = self.detector.detect_faces(frame, person_bbox)
                for face_info in faces:
                    all_faces.append((
                        face_info['face_img'],
                        face_info['bbox'],
                        person_id,
                        person_conf,
                        face_info['face_obj'],
                        face_info['confidence']
                    ))
        else:
            # If no person tracking, detect faces in the whole frame
            faces = self.detector.detect_faces(frame)
            for face_info in faces:
                all_faces.append((
                    face_info['face_img'],
                    face_info['bbox'],
                    None,
                    None,
                    face_info['face_obj'],
                    face_info['confidence']
                ))
        return all_faces
    
    def get_face_stats(self) -> Dict:
        """Get face detection statistics"""
        return {
            'frame_count': self.frame_count,
            'detection_interval': FACE_DETECTION_INTERVAL
        } 
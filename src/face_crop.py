"""
Face detection and cropping within person bounding boxes using InsightFace
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import insightface
from insightface.app import FaceAnalysis
from src.config import *

class FaceDetector:
    """InsightFace face detector"""
    def __init__(self):
        """Initialize InsightFace face detector"""
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
    def detect_faces(self, frame: np.ndarray, person_bbox: List[int]) -> List[Dict]:
        """Detect faces within a person bounding box"""
        x1, y1, x2, y2 = person_bbox
        
        # Crop person region
        person_region = frame[y1:y2, x1:x2]
        if person_region.size == 0:
            return []
        
        # Detect faces in the person region
        faces = self.app.get(person_region)
        
        face_detections = []
        for face in faces:
            # Get face bounding box
            bbox = face.bbox.astype(int)
            
            # Adjust coordinates to original frame
            bbox[0] += x1
            bbox[1] += y1
            bbox[2] += x1
            bbox[3] += y1
            
            # Check minimum face size
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            
            if face_width >= MIN_FACE_SIZE and face_height >= MIN_FACE_SIZE:
                # Crop face image
                face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # Ensure face image is valid
                if face_img.size > 0 and face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    face_detections.append({
                        'bbox': bbox.tolist(),
                        'face_img': face_img,
                        'face_obj': face,
                        'confidence': face.det_score
                    })
        
        return face_detections

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
        
        for person_bbox, person_id, person_conf in tracked_persons:
            # Detect faces in this person
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
        
        return all_faces
    
    def get_face_stats(self) -> Dict:
        """Get face detection statistics"""
        return {
            'frame_count': self.frame_count,
            'detection_interval': FACE_DETECTION_INTERVAL
        } 
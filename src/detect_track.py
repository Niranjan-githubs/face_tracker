"""
Person detection and tracking using YOLOv8 and DeepSORT
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import torch
from src.config import *
from src.utils import calculate_iou

class DeepSORTTracker:
    """
    Simplified DeepSORT implementation for person tracking
    """
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
        
    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """Update tracks with new detections using DeepSORT logic"""
        if not detections:
            # Update existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for bbox, conf in detections:
                track_id = self.next_id
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'confidence': conf,
                    'disappeared': 0,
                    'hits': 1
                }
                self.next_id += 1
            return [(bbox, track_id, conf) for track_id, (bbox, conf) in enumerate(detections, 1)]
        
        # Match detections to existing tracks using IoU
        matched_tracks = set()
        current_tracks = []
        
        for bbox, conf in detections:
            best_match = None
            best_iou = self.iou_threshold
            
            for track_id, track_info in self.tracks.items():
                if track_info['disappeared'] <= self.max_disappeared:
                    iou = calculate_iou(bbox, track_info['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.tracks[best_match]['bbox'] = bbox
                self.tracks[best_match]['confidence'] = conf
                self.tracks[best_match]['disappeared'] = 0
                self.tracks[best_match]['hits'] += 1
                matched_tracks.add(best_match)
                current_tracks.append((bbox, best_match, conf))
            else:
                # Create new track
                track_id = self.next_id
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'confidence': conf,
                    'disappeared': 0,
                    'hits': 1
                }
                current_tracks.append((bbox, track_id, conf))
                self.next_id += 1
        
        # Update disappeared count for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]['disappeared'] += 1
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
        
        return current_tracks

class PersonDetector:
    """YOLOv8 person detector"""
    def __init__(self):
        """Initialize YOLOv8 person detector"""
        self.model = YOLO('yolov8n.pt')  # Use nano model for CPU
        self.device = 'cpu'
        
    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """Detect people in frame"""
        results = self.model(frame, classes=[0])  # class 0 is person
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > PERSON_CONFIDENCE_THRESHOLD:
                        detections.append(([int(x1), int(y1), int(x2), int(y2)], confidence))
        
        return detections

class PersonTracker:
    """Complete person detection and tracking system"""
    def __init__(self):
        """Initialize person detection and tracking system"""
        self.detector = PersonDetector()
        self.tracker = DeepSORTTracker()
        
    def process_frame(self, frame: np.ndarray) -> List[Tuple]:
        """Process frame and return tracked persons"""
        # Detect people
        detections = self.detector.detect(frame)
        
        # Track people using DeepSORT
        tracks = self.tracker.update(detections)
        
        return tracks
    
    def get_track_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.tracker.tracks),
            'total_tracks': self.tracker.next_id - 1
        } 
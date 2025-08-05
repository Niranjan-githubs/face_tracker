"""
Person detection and tracking using YOLOv8 and ByteTracker
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import torch
from src.config import *
from src.utils import calculate_iou

class ByteTracker:
    """
    Simplified ByteTracker implementation for person tracking
    """
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.track_buffer = TRACK_BUFFER
        self.track_thresh = TRACK_THRESH
        self.high_thresh = HIGH_THRESH
        self.match_thresh = MATCH_THRESH
        
    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """Update tracks with new detections using ByteTracker logic"""
        if not detections:
            # Update existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['time_since_update'] += 1
                if self.tracks[track_id]['time_since_update'] > self.track_buffer:
                    del self.tracks[track_id]
            return []
        
        # Separate high and low confidence detections
        high_conf_dets = []
        low_conf_dets = []
        
        for bbox, conf in detections:
            if conf >= self.high_thresh:
                high_conf_dets.append((bbox, conf))
            elif conf >= self.track_thresh:
                low_conf_dets.append((bbox, conf))
        
        # First, match high confidence detections to existing tracks
        matched_tracks = set()
        current_tracks = []
        
        for bbox, conf in high_conf_dets:
            best_match = None
            best_iou = self.match_thresh
            
            for track_id, track_info in self.tracks.items():
                if track_info['time_since_update'] <= self.track_buffer:
                    iou = calculate_iou(bbox, track_info['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.tracks[best_match]['bbox'] = bbox
                self.tracks[best_match]['confidence'] = conf
                self.tracks[best_match]['time_since_update'] = 0
                matched_tracks.add(best_match)
                current_tracks.append((bbox, best_match, conf))
            else:
                # Create new track
                track_id = self.next_id
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'confidence': conf,
                    'time_since_update': 0
                }
                current_tracks.append((bbox, track_id, conf))
                self.next_id += 1
        
        # Then, try to match low confidence detections to unmatched tracks
        for bbox, conf in low_conf_dets:
            best_match = None
            best_iou = self.match_thresh
            
            for track_id, track_info in self.tracks.items():
                if (track_id not in matched_tracks and 
                    track_info['time_since_update'] <= self.track_buffer):
                    iou = calculate_iou(bbox, track_info['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.tracks[best_match]['bbox'] = bbox
                self.tracks[best_match]['confidence'] = conf
                self.tracks[best_match]['time_since_update'] = 0
                matched_tracks.add(best_match)
                current_tracks.append((bbox, best_match, conf))
        
        # Update time_since_update for unmatched tracks
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]['time_since_update'] += 1
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['time_since_update'] > self.track_buffer:
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
        self.tracker = ByteTracker()
        
    def process_frame(self, frame: np.ndarray) -> List[Tuple]:
        """Process frame and return tracked persons"""
        # Detect people
        detections = self.detector.detect(frame)
        
        # Track people using ByteTracker
        tracks = self.tracker.update(detections)
        
        return tracks
    
    def get_track_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.tracker.tracks),
            'total_tracks': self.tracker.next_id - 1
        } 
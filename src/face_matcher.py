"""
Enhanced Face Matcher with Cosine Similarity and Persistent IDs
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import time
from src.utils import cosine_similarity
from src.config import *

class EnhancedFaceMatcher:
    """Advanced face matching system with persistent IDs and cosine similarity"""
    
    def __init__(self):
        """Initialize the enhanced face matcher"""
        # Persistent face storage: {persistent_id: {'embedding': array, 'first_seen': int, 'last_seen': int, 'count': int}}
        self.persistent_faces = {}
        self.next_persistent_id = 1
        
        # Frame-level tracking: {frame_id: {face_id: persistent_id}}
        self.frame_tracking = {}
        self.current_frame_id = 0
        
        # Performance metrics
        self.total_matches = 0
        self.new_faces = 0
        self.reappearances = 0
        
        # Temporal tracking for reappearance detection
        self.disappeared_faces = {}  # {persistent_id: {'last_seen': int, 'embedding': array}}
        self.reappearance_threshold = 30  # frames to consider as reappearance
        
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeepFace ArcFace model"""
        try:
            from deepface import DeepFace
            import cv2
            # Convert BGR to RGB (DeepFace expects RGB)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # Get embedding using DeepFace ArcFace
            embedding = DeepFace.represent(
                img_path=face_rgb,
                model_name="ArcFace",
                enforce_detection=False
            )
            if embedding is None:
                print("ArcFace embedding failed for this crop!")
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
        except Exception as e:
            print(f"Error extracting embedding: {e}")
        return None
    
    def find_best_match(self, embedding: np.ndarray, threshold: float = 0.7) -> Tuple[Optional[int], float]:
        """Find best matching persistent face using cosine similarity"""
        best_match_id = None
        best_similarity = 0.0
        
        # Check active persistent faces
        for persistent_id, face_data in self.persistent_faces.items():
            similarity = cosine_similarity(embedding, face_data['embedding'])
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match_id = persistent_id
        
        # Check disappeared faces for reappearance
        for persistent_id, face_data in self.disappeared_faces.items():
            if self.current_frame_id - face_data['last_seen'] > self.reappearance_threshold:
                similarity = cosine_similarity(embedding, face_data['embedding'])
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match_id = persistent_id
        
        return best_match_id, best_similarity
    
    def add_persistent_face(self, embedding: np.ndarray) -> int:
        """Add new face to persistent storage"""
        persistent_id = self.next_persistent_id
        self.persistent_faces[persistent_id] = {
            'embedding': embedding,
            'first_seen': self.current_frame_id,
            'last_seen': self.current_frame_id,
            'count': 1
        }
        self.next_persistent_id += 1
        self.new_faces += 1
        return persistent_id
    
    def update_persistent_face(self, persistent_id: int):
        """Update persistent face with new sighting"""
        if persistent_id in self.persistent_faces:
            self.persistent_faces[persistent_id]['last_seen'] = self.current_frame_id
            self.persistent_faces[persistent_id]['count'] += 1
        elif persistent_id in self.disappeared_faces:
            # Reappearance detected
            face_data = self.disappeared_faces[persistent_id]
            self.persistent_faces[persistent_id] = {
                'embedding': face_data['embedding'],
                'first_seen': face_data['first_seen'],
                'last_seen': self.current_frame_id,
                'count': face_data['count'] + 1
            }
            del self.disappeared_faces[persistent_id]
            self.reappearances += 1
    
    def mark_face_disappeared(self, persistent_id: int):
        """Mark a face as disappeared"""
        if persistent_id in self.persistent_faces:
            face_data = self.persistent_faces[persistent_id].copy()
            face_data['last_seen'] = self.current_frame_id
            self.disappeared_faces[persistent_id] = face_data
            del self.persistent_faces[persistent_id]
    
    def process_frame_faces(self, frame_faces: List[Tuple]) -> List[Tuple]:
        """Process faces in current frame and assign persistent IDs"""
        self.current_frame_id += 1
        processed_faces = []
        current_frame_matches = {}
        
        for face_img, face_bbox, person_id, person_conf, face_obj, face_conf in frame_faces:
            # Extract embedding
            embedding = self.extract_embedding(face_img)
            
            if embedding is not None:
                # Find best match
                persistent_id, similarity = self.find_best_match(embedding, FACE_SIMILARITY_THRESHOLD)
                
                if persistent_id is not None:
                    # Found matching persistent face
                    self.update_persistent_face(persistent_id)
                    confidence = similarity
                    self.total_matches += 1
                else:
                    # New persistent face
                    persistent_id = self.add_persistent_face(embedding)
                    confidence = 1.0
                
                current_frame_matches[person_id] = persistent_id
                
                processed_faces.append((
                    face_img,
                    face_bbox,
                    person_id,
                    persistent_id,
                    confidence,
                    person_conf,
                    face_conf
                ))
        
        # Track frame-level matches
        self.frame_tracking[self.current_frame_id] = current_frame_matches
        
        # Clean up old disappeared faces
        self._cleanup_disappeared_faces()
        
        return processed_faces
    
    def _cleanup_disappeared_faces(self):
        """Remove faces that have been disappeared for too long"""
        current_time = self.current_frame_id
        to_remove = []
        
        for persistent_id, face_data in self.disappeared_faces.items():
            if current_time - face_data['last_seen'] > 300:  # 10 seconds at 30fps
                to_remove.append(persistent_id)
        
        for persistent_id in to_remove:
            del self.disappeared_faces[persistent_id]
    
    def get_persistent_face_info(self, persistent_id: int) -> Optional[Dict]:
        """Get information about a specific persistent face"""
        if persistent_id in self.persistent_faces:
            return self.persistent_faces[persistent_id]
        elif persistent_id in self.disappeared_faces:
            return self.disappeared_faces[persistent_id]
        return None
    
    def get_statistics(self) -> Dict:
        """Get comprehensive matching statistics"""
        active_faces = len(self.persistent_faces)
        disappeared_faces = len(self.disappeared_faces)
        total_unique = self.next_persistent_id - 1
        
        return {
            'active_persistent_faces': active_faces,
            'disappeared_faces': disappeared_faces,
            'total_unique_faces': total_unique,
            'total_matches': self.total_matches,
            'new_faces_detected': self.new_faces,
            'reappearances': self.reappearances,
            'current_frame': self.current_frame_id
        }
    
    def get_face_history(self, persistent_id: int) -> List[Tuple]:
        """Get frame history for a specific persistent face"""
        history = []
        for frame_id, frame_matches in self.frame_tracking.items():
            for person_id, p_id in frame_matches.items():
                if p_id == persistent_id:
                    history.append((frame_id, person_id))
        return sorted(history)
    
    def clear_all_data(self):
        """Clear all persistent data"""
        self.persistent_faces.clear()
        self.disappeared_faces.clear()
        self.frame_tracking.clear()
        self.next_persistent_id = 1
        self.current_frame_id = 0
        self.total_matches = 0
        self.new_faces = 0
        self.reappearances = 0 
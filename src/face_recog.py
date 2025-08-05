"""
Face recognition and re-identification using InsightFace embeddings
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import insightface
from insightface.app import FaceAnalysis
from src.utils import cosine_similarity
from src.config import *

class FaceRecognizer:
    """Face recognition and re-identification system"""
    def __init__(self):
        """Initialize face recognition system"""
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Embedding cache: {face_id: {'embedding': array, 'count': int, 'last_seen': int}}
        self.embedding_cache = {}
        self.next_face_id = 1
        self.frame_count = 0
        self.matches_count = 0
        
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image"""
        try:
            # Get face embedding
            faces = self.app.get(face_img)
            if faces:
                return faces[0].embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
        return None
    
    def find_matching_face(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Find matching face in cache using cosine similarity"""
        best_match_id = None
        best_similarity = 0.0
        
        for face_id, face_data in self.embedding_cache.items():
            similarity = cosine_similarity(embedding, face_data['embedding'])
            
            if similarity > best_similarity and similarity >= FACE_SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match_id = face_id
        
        return best_match_id, best_similarity
    
    def add_face_to_cache(self, embedding: np.ndarray) -> int:
        """Add new face to cache and return face ID"""
        face_id = self.next_face_id
        self.embedding_cache[face_id] = {
            'embedding': embedding,
            'count': 1,
            'last_seen': self.frame_count
        }
        self.next_face_id += 1
        
        # Limit cache size
        if len(self.embedding_cache) > EMBEDDING_CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_id = min(self.embedding_cache.keys())
            del self.embedding_cache[oldest_id]
        
        return face_id
    
    def update_face_count(self, face_id: int):
        """Update appearance count for a face"""
        if face_id in self.embedding_cache:
            self.embedding_cache[face_id]['count'] += 1
            self.embedding_cache[face_id]['last_seen'] = self.frame_count
    
    def process_faces(self, detected_faces: List[Tuple]) -> List[Tuple]:
        """Process detected faces for re-identification"""
        self.frame_count += 1
        processed_faces = []
        
        for face_img, face_bbox, person_id, person_conf, face_obj, face_conf in detected_faces:
            # Extract embedding
            embedding = self.extract_embedding(face_img)
            
            if embedding is not None:
                # Find matching face
                match_id, similarity = self.find_matching_face(embedding)
                
                if match_id is not None:
                    # Found matching face
                    self.update_face_count(match_id)
                    face_id = match_id
                    confidence = similarity
                    self.matches_count += 1
                else:
                    # New face
                    face_id = self.add_face_to_cache(embedding)
                    confidence = 1.0
                
                processed_faces.append((
                    face_img,
                    face_bbox,
                    person_id,
                    face_id,
                    confidence,
                    person_conf,
                    face_conf
                ))
        
        return processed_faces
    
    def get_face_stats(self) -> Dict:
        """Get statistics about recognized faces"""
        return {
            'total_faces': len(self.embedding_cache),
            'unique_faces': self.next_face_id - 1,
            'embedding_matches': self.matches_count,
            'cache_size': len(self.embedding_cache)
        }
    
    def get_face_info(self, face_id: int) -> Optional[Dict]:
        """Get information about a specific face"""
        if face_id in self.embedding_cache:
            return self.embedding_cache[face_id]
        return None
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.next_face_id = 1
        self.matches_count = 0 
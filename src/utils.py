"""
Utility functions for the real-time video analytics system
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Optional
from collections import deque
from src.config import *

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(a - b)

def draw_bounding_box(frame: np.ndarray, bbox: List[int], 
                     label: str, color: Tuple[int, int, int], 
                     confidence: Optional[float] = None,
                     thickness: int = 2) -> np.ndarray:
    """Draw bounding box with label and optional confidence"""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Create label text
    label_text = label
    if confidence is not None and SHOW_CONFIDENCE_SCORES:
        label_text += f" ({confidence:.2f})"
    
    # Calculate text position
    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1]
    
    # Draw text background
    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), 
                 (text_x + text_size[0], text_y + 5), color, -1)
    cv2.putText(frame, label_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def draw_stats(frame: np.ndarray, stats: Dict) -> np.ndarray:
    """Draw statistics on frame"""
    if not SHOW_STATS:
        return frame
    
    y_offset = 30
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        y_offset += 30
    
    return frame

def create_face_panel(faces: List[Tuple], max_faces: int = None) -> Optional[np.ndarray]:
    """Create a panel showing detected faces"""
    if not faces or not SHOW_FACE_PANELS:
        return None
    
    if max_faces is None:
        max_faces = MAX_FACES_IN_PANEL
    
    # Limit number of faces to display
    faces = faces[:max_faces]
    
    # Calculate panel dimensions
    face_size = FACE_PANEL_SIZE
    cols = min(4, len(faces))
    rows = (len(faces) + cols - 1) // cols
    
    panel = np.full((rows * face_size + 20, cols * face_size + 20, 3), 
                   PANEL_BG_COLOR, dtype=np.uint8)
    
    for i, (face_img, face_id, confidence) in enumerate(faces):
        if face_img is None:
            continue
            
        # Resize face image
        face_resized = cv2.resize(face_img, (face_size, face_size))
        
        # Calculate position
        row = i // cols
        col = i % cols
        x = col * face_size + 10
        y = row * face_size + 10
        
        # Place face in panel
        panel[y:y+face_size, x:x+face_size] = face_resized
        
        # Add ID label
        label = f"ID:{face_id}"
        if confidence is not None:
            label += f"({confidence:.2f})"
        
        cv2.putText(panel, label, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return panel

def ensure_directory(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def get_fps(start_time: float, frame_count: int) -> float:
    """Calculate current FPS"""
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def resize_image(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    if max(height, width) <= max_size:
        return image
    
    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height))

def save_face_image(face_img: np.ndarray, face_id: int, frame_count: int):
    """Save individual face image"""
    ensure_directory(FACES_DIR)
    filename = f"face_{face_id}_frame_{frame_count}.jpg"
    filepath = os.path.join(FACES_DIR, filename)
    cv2.imwrite(filepath, face_img)

class FPS:
    """FPS counter utility"""
    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.fps_times = deque(maxlen=avg_frames)
        self.fps = 0.0
        
    def update(self):
        """Update FPS calculation"""
        self.fps_times.append(time.time())
        if len(self.fps_times) > 1:
            self.fps = len(self.fps_times) / (self.fps_times[-1] - self.fps_times[0])
        return self.fps
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps 
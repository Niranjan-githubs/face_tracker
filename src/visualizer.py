"""
Live Dashboard Visualizer with Real-time Statistics and Face Panels
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
from src.config import *

class LiveDashboard:
    """Real-time dashboard for face tracking visualization"""
    
    def __init__(self):
        """Initialize the live dashboard"""
        # Dashboard dimensions
        self.dashboard_width = 1200
        self.dashboard_height = 800
        self.main_video_width = 800
        self.main_video_height = 600
        
        # Statistics panel
        self.stats_panel_width = 400
        self.stats_panel_height = 600
        
        # Face panel
        self.face_panel_width = 400
        self.face_panel_height = 200
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Statistics
        self.stats = {
            'FPS': 0,
            'Total People': 0,
            'Active IDs': 0,
            'Faces Detected': 0,
            'Unique Faces': 0,
            'Matches': 0,
            'Processing Time': 0
        }
        
        # Face thumbnails storage
        self.face_thumbnails = {}  # {persistent_id: thumbnail_img}
        self.max_thumbnails = 12
        
        # Colors (BGR format)
        self.colors = {
            'background': (40, 40, 40),
            'panel_bg': (60, 60, 60),
            'text': (255, 255, 255),
            'highlight': (0, 255, 0),
            'warning': (0, 165, 255),
            'error': (0, 0, 255)
        }
    
    def create_dashboard(self, main_frame: np.ndarray, 
                        tracked_persons: List, 
                        processed_faces: List,
                        stats: Dict) -> np.ndarray:
        """Create complete dashboard with main video, statistics, and face panel"""
        # Create dashboard canvas
        dashboard = np.full((self.dashboard_height, self.dashboard_width, 3), 
                          self.colors['background'], dtype=np.uint8)
        
        # Update statistics
        self.update_stats(stats)
        
        # Create main video panel
        main_panel = self.create_main_video_panel(main_frame, tracked_persons, processed_faces)
        
        # Create statistics panel
        stats_panel = self.create_stats_panel()
        
        # Create face thumbnails panel
        face_panel = self.create_face_panel(processed_faces)
        
        # Combine panels
        # Main video (top left)
        dashboard[0:self.main_video_height, 0:self.main_video_width] = main_panel
        
        # Statistics panel (top right)
        stats_y = 0
        stats_x = self.main_video_width
        dashboard[stats_y:stats_y+self.stats_panel_height, 
                 stats_x:stats_x+self.stats_panel_width] = stats_panel
        
        # Face panel (bottom)
        face_y = self.main_video_height
        face_x = 0
        dashboard[face_y:face_y+self.face_panel_height, 
                 face_x:face_x+self.dashboard_width] = face_panel
        
        return dashboard
    
    def create_main_video_panel(self, frame: np.ndarray, 
                               tracked_persons: List, 
                               processed_faces: List) -> np.ndarray:
        """Create main video panel with bounding boxes, IDs, and analytics overlay"""
        # Resize frame to fit panel
        panel = cv2.resize(frame, (self.main_video_width, self.main_video_height))
        
        # Draw person bounding boxes
        for bbox, person_id, confidence in tracked_persons:
            x1, y1, x2, y2 = bbox
            # Scale coordinates
            x1 = int(x1 * self.main_video_width / frame.shape[1])
            y1 = int(y1 * self.main_video_height / frame.shape[0])
            x2 = int(x2 * self.main_video_width / frame.shape[1])
            y2 = int(y2 * self.main_video_height / frame.shape[0])
            
            # Draw box
            cv2.rectangle(panel, (x1, y1), (x2, y2), PERSON_BOX_COLOR, 2)
            
            # Draw ID and confidence
            label = f"Person {person_id}"
            if SHOW_CONFIDENCE_SCORES:
                label += f" ({confidence:.2f})"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(panel, (x1, y1-label_height-10), (x1+label_width, y1), PERSON_BOX_COLOR, -1)
            
            # Draw label text
            cv2.putText(panel, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Draw face bounding boxes
        for face_img, face_bbox, person_id, persistent_id, confidence, person_conf, face_conf in processed_faces:
            x1, y1, x2, y2 = face_bbox
            # Scale coordinates
            x1 = int(x1 * self.main_video_width / frame.shape[1])
            y1 = int(y1 * self.main_video_height / frame.shape[0])
            x2 = int(x2 * self.main_video_width / frame.shape[1])
            y2 = int(y2 * self.main_video_height / frame.shape[0])
            
            # Draw face box
            cv2.rectangle(panel, (x1, y1), (x2, y2), FACE_BOX_COLOR, 2)
            
            # Draw persistent ID
            label = f"Face {persistent_id}"
            if SHOW_CONFIDENCE_SCORES:
                label += f" ({confidence:.2f})"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(panel, (x1, y2+5), (x1+label_width, y2+label_height+10), FACE_BOX_COLOR, -1)
            
            # Draw label text
            cv2.putText(panel, label, (x1, y2+label_height+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Add analytics overlay
        self.draw_analytics_overlay(panel, tracked_persons, processed_faces)
        
        return panel
    
    def draw_analytics_overlay(self, panel: np.ndarray, tracked_persons: List, processed_faces: List):
        """Draw analytics overlay on the video panel"""
        # Create semi-transparent overlay for analytics
        overlay = panel.copy()
        
        # Analytics box background
        box_x, box_y = 10, 10
        box_width, box_height = 300, 120
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     self.colors['highlight'], 2)
        
        # Analytics text
        y_offset = box_y + 25
        line_height = 20
        
        # Title
        cv2.putText(overlay, "LIVE ANALYTICS", (box_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['highlight'], 2)
        y_offset += line_height + 5
        
        # Statistics
        stats_text = [
            f"People: {len(tracked_persons)}",
            f"Faces: {len(processed_faces)}",
            f"FPS: {self.stats['FPS']:.1f}",
            f"Frame: {self.stats.get('Total Frames', 0)}"
        ]
        
        for text in stats_text:
            cv2.putText(overlay, text, (box_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += line_height
        
        # Apply transparency
        alpha = 0.7
        panel = cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(panel, timestamp, (panel.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def create_stats_panel(self) -> np.ndarray:
        """Create real-time statistics panel"""
        panel = np.full((self.stats_panel_height, self.stats_panel_width, 3), 
                       self.colors['panel_bg'], dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "REAL-TIME STATISTICS", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['highlight'], 2)
        
        # Draw statistics
        y_offset = 80
        line_height = 35
        
        stats_to_display = [
            ('FPS', self.stats['FPS'], '{:.1f}'),
            ('Total People', self.stats['Total People'], '{}'),
            ('Active IDs', self.stats['Active IDs'], '{}'),
            ('Faces Detected', self.stats['Faces Detected'], '{}'),
            ('Unique Faces', self.stats['Unique Faces'], '{}'),
            ('Matches', self.stats['Matches'], '{}'),
            ('Processing Time', self.stats['Processing Time'], '{:.2f}ms')
        ]
        
        for label, value, format_str in stats_to_display:
            # Label
            cv2.putText(panel, label, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            
            # Value
            value_str = format_str.format(value)
            cv2.putText(panel, value_str, (200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['highlight'], 2)
            
            y_offset += line_height
        
        # Performance graph
        if len(self.fps_history) > 1:
            self.draw_performance_graph(panel, 20, y_offset + 20, 360, 100)
        
        return panel
    
    def create_face_panel(self, processed_faces: List) -> np.ndarray:
        """Create face thumbnails panel with IDs"""
        panel = np.full((self.face_panel_height, self.dashboard_width, 3), 
                       self.colors['panel_bg'], dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "FACE THUMBNAILS WITH IDs", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['highlight'], 2)
        
        if not processed_faces:
            cv2.putText(panel, "No faces detected", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            return panel
        
        # Update face thumbnails
        self.update_face_thumbnails(processed_faces)
        
        # Display thumbnails
        thumbnail_size = 80
        thumbnails_per_row = self.dashboard_width // (thumbnail_size + 20)
        
        y_start = 50
        for i, (persistent_id, thumbnail) in enumerate(self.face_thumbnails.items()):
            if i >= self.max_thumbnails:
                break
            
            row = i // thumbnails_per_row
            col = i % thumbnails_per_row
            
            x = 20 + col * (thumbnail_size + 20)
            y = y_start + row * (thumbnail_size + 30)
            
            # Resize thumbnail
            thumbnail_resized = cv2.resize(thumbnail, (thumbnail_size, thumbnail_size))
            
            # Add border
            thumbnail_with_border = cv2.copyMakeBorder(thumbnail_resized, 2, 2, 2, 2, 
                                                     cv2.BORDER_CONSTANT, value=self.colors['highlight'])
            
            # Ensure the region fits within panel bounds
            end_y = min(y + thumbnail_size + 4, panel.shape[0])
            end_x = min(x + thumbnail_size + 4, panel.shape[1])
            actual_height = end_y - y
            actual_width = end_x - x
            
            # Crop thumbnail_with_border if needed
            thumbnail_cropped = thumbnail_with_border[:actual_height, :actual_width]
            
            # Place thumbnail
            panel[y:end_y, x:end_x] = thumbnail_cropped
            
            # Add ID label
            label = f"ID {persistent_id}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            # Label background
            cv2.rectangle(panel, (x, y+thumbnail_size+4), 
                         (x+label_width+4, y+thumbnail_size+4+label_height+4), 
                         self.colors['panel_bg'], -1)
            
            # Label text
            cv2.putText(panel, label, (x+2, y+thumbnail_size+4+label_height+2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return panel
    
    def update_face_thumbnails(self, processed_faces: List):
        """Update face thumbnails with latest detections"""
        for face_img, face_bbox, person_id, persistent_id, confidence, person_conf, face_conf in processed_faces:
            # Store thumbnail
            self.face_thumbnails[persistent_id] = face_img.copy()
            
            # Limit number of thumbnails
            if len(self.face_thumbnails) > self.max_thumbnails:
                # Remove oldest thumbnail
                oldest_id = min(self.face_thumbnails.keys())
                del self.face_thumbnails[oldest_id]
    
    def draw_performance_graph(self, panel: np.ndarray, x: int, y: int, width: int, height: int):
        """Draw FPS performance graph"""
        if len(self.fps_history) < 2:
            return
        
        # Create graph background
        cv2.rectangle(panel, (x, y), (x+width, y+height), self.colors['background'], -1)
        
        # Draw grid lines
        for i in range(5):
            grid_y = y + (i * height // 4)
            cv2.line(panel, (x, grid_y), (x+width, grid_y), (80, 80, 80), 1)
        
        # Draw FPS line
        fps_values = list(self.fps_history)
        if len(fps_values) > 1:
            max_fps = max(fps_values) if fps_values else 30
            min_fps = min(fps_values) if fps_values else 0
            
            points = []
            for i, fps in enumerate(fps_values):
                graph_x = x + int((i / (len(fps_values) - 1)) * width)
                if max_fps > min_fps:
                    graph_y = y + height - int(((fps - min_fps) / (max_fps - min_fps)) * height)
                else:
                    graph_y = y + height // 2
                points.append((graph_x, graph_y))
            
            # Draw line
            for i in range(len(points) - 1):
                cv2.line(panel, points[i], points[i+1], self.colors['highlight'], 2)
    
    def update_stats(self, new_stats: Dict):
        """Update statistics with new values"""
        self.stats.update(new_stats)
        
        # Update FPS
        current_time = time.time()
        if self.last_frame_time > 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.stats['FPS'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                self.fps_history.append(self.stats['FPS'])
        
        self.last_frame_time = current_time
    
    def show_dashboard(self, dashboard: np.ndarray):
        """Display the dashboard"""
        cv2.imshow('Face Tracking Dashboard', dashboard)
    
    def get_dashboard_size(self) -> Tuple[int, int]:
        """Get dashboard dimensions"""
        return (self.dashboard_width, self.dashboard_height) 
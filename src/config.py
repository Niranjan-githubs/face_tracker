"""
Configuration file for the real-time video analytics system
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
FACES_DIR = os.path.join(ASSETS_DIR, "faces")

# Video Processing
INPUT_VIDEO = os.path.join(ASSETS_DIR, "input_video.mp4")
OUTPUT_VIDEO = os.path.join(ASSETS_DIR, "output_video.mp4")

# Performance Settings
TARGET_FPS = 15
FACE_DETECTION_INTERVAL = 1  # Detect faces every frame to reduce flickering
MIN_FACE_SIZE = 30

# Detection & Tracking
PERSON_CONFIDENCE_THRESHOLD = 0.5
FACE_CONFIDENCE_THRESHOLD = 0.7
TRACKING_ALGORITHM = "deepsort"  # "bytetrack" or "deepsort"

# Face Re-Identification
FACE_SIMILARITY_THRESHOLD = 0.5  # Lower threshold for better matching
EMBEDDING_CACHE_SIZE = 1000  # Maximum number of embeddings to store
SHOW_CONFIDENCE_SCORES = True

# Visualization
DRAW_BOUNDING_BOXES = True
DRAW_TRACK_IDS = True
SHOW_FACE_PANELS = True
SHOW_STATS = True
FACE_PANEL_SIZE = 80
MAX_FACES_IN_PANEL = 8

# Colors (BGR format)
PERSON_BOX_COLOR = (0, 255, 0)  # Green
FACE_BOX_COLOR = (255, 0, 0)    # Blue
TEXT_COLOR = (255, 255, 255)    # White
PANEL_BG_COLOR = (50, 50, 50)   # Dark gray

# Model paths
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
FACENET_MODEL_PATH = os.path.join(MODELS_DIR, "buffalo_l")

# DeepSORT settings
MAX_DISAPPEARED = 30
MIN_HITS = 3
IOU_THRESHOLD = 0.3 
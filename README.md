# Real-Time Video Analytics System

A comprehensive real-time video analytics system that can automatically detect, track, and identify people in any video using only free, open-source models.

## 🎯 Features

- **Person Detection**: YOLOv8 for accurate person detection
- **Person Tracking**: ByteTracker for robust multi-object tracking
- **Face Detection**: InsightFace RetinaFace for precise face detection
- **Face Re-identification**: 512D embeddings with cosine similarity matching
- **Real-time Visualization**: Bounding boxes, IDs, confidence scores, face panels
- **Performance Monitoring**: FPS, statistics, and progress tracking
- **Video Export**: Processed video with overlays and annotations

## 🏗️ System Architecture

```
realtime_video_analytics/
├── src/
│   ├── detect_track.py    # Person detection & tracking (YOLOv8 + ByteTracker)
│   ├── face_crop.py       # Face detection within person boxes
│   ├── face_recog.py      # Face recognition & re-identification
│   ├── utils.py           # Utility functions & helpers
│   ├── config.py          # Configuration settings
│   └── main.py            # Main processing pipeline
├── models/                # Model files (auto-downloaded)
├── assets/
│   ├── input_video.mp4    # Input video file
│   ├── output_video.avi   # Processed output video
│   └── faces/             # Cropped face images
├── output/                # Additional outputs
└── requirements.txt       # Python dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd realtime_video_analytics
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your video**:
   - Place your MP4 video file in the `assets/` directory
   - Rename it to `input_video.mp4` or update the path in `src/config.py`

## 🎮 Usage

### Basic Usage

Run the system with default settings:
```bash
python src/main.py
```

### Interactive Controls

During processing, you can use these keyboard controls:
- **`q`**: Quit the application
- **`s`**: Save current frame as image
- **`c`**: Clear face recognition cache

### Configuration

Edit `src/config.py` to customize:
- Detection thresholds
- Tracking parameters
- Face recognition settings
- Visualization options
- Performance settings

## ⚙️ Configuration Options

### Detection & Tracking
```python
PERSON_CONFIDENCE_THRESHOLD = 0.5    # Person detection confidence
FACE_CONFIDENCE_THRESHOLD = 0.7      # Face detection confidence
TRACKING_ALGORITHM = "bytetrack"     # Tracking algorithm
```

### Face Re-identification
```python
FACE_SIMILARITY_THRESHOLD = 0.7      # Face matching threshold
EMBEDDING_CACHE_SIZE = 1000          # Max embeddings to store
```

### Performance
```python
TARGET_FPS = 15                      # Target processing FPS
FACE_DETECTION_INTERVAL = 3          # Face detection frequency
```

## 🔧 Technical Details

### Person Detection & Tracking
- **Model**: YOLOv8-nano (optimized for CPU)
- **Algorithm**: ByteTracker for robust tracking
- **Features**: Handles occlusion, reappearance, and ID persistence

### Face Detection & Recognition
- **Model**: InsightFace (RetinaFace + ArcFace)
- **Embedding**: 512-dimensional face embeddings
- **Matching**: Cosine similarity with configurable threshold
- **Cache**: FIFO cache for efficient memory usage

### Performance Optimizations
- Frame skipping for face detection
- Efficient bounding box calculations
- Memory-managed embedding cache
- CPU-optimized model selection

## 📊 Output

### Video Output
- Processed video with bounding boxes and IDs
- Face panels showing detected faces
- Real-time statistics overlay
- Saved to `assets/output_video.avi`

### Face Images
- Individual face crops saved to `assets/faces/`
- Named with face ID and frame number
- Useful for analysis and verification

### Statistics
- FPS monitoring
- Person and face counts
- Unique face identification
- Embedding match statistics

## 🎯 Use Cases

- **Security Monitoring**: Track people across camera feeds
- **Retail Analytics**: Customer behavior analysis
- **Event Management**: Attendee tracking and counting
- **Research**: Human behavior studies
- **Content Creation**: Video annotation and analysis

## 🔍 Troubleshooting

### Common Issues

1. **Low FPS**: 
   - Reduce `FACE_DETECTION_INTERVAL`
   - Use smaller input video resolution
   - Consider GPU acceleration

2. **Poor Face Recognition**:
   - Adjust `FACE_SIMILARITY_THRESHOLD`
   - Increase `MIN_FACE_SIZE`
   - Improve video quality

3. **Memory Issues**:
   - Reduce `EMBEDDING_CACHE_SIZE`
   - Process shorter video segments
   - Clear cache periodically

### Performance Tips

- Use 720p or lower resolution for real-time processing
- Adjust detection intervals based on your hardware
- Monitor system resources during processing
- Consider batch processing for large videos

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New features
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **ByteTracker**: ByteDance for tracking algorithm
- **InsightFace**: DeepInsight for face recognition
- **OpenCV**: Computer vision library

---

**Note**: This system uses only free, open-source models and libraries. No paid APIs or services are required. 
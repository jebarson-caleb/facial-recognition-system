# Performance Guide for Fast Face & Emotion Detection

## Quick Setup Tips

### 1. Optimal Camera Position
- Position yourself 2-3 feet from the camera
- Look directly at the camera for best detection
- Avoid rapid head movements for stable tracking

### 2. Lighting Recommendations
- Ensure your face is well-lit (avoid backlighting)
- Avoid harsh shadows on your face
- Ensure consistent lighting (avoid flickering lights)
- Natural daylight works best

### 3. System Performance
- Close other camera applications before running

### 4. Fine-tuning Parameters
You can adjust these values in `config.ini` for your specific setup:
- `scale_factor`: Adjusts the image scale for detection. A smaller value increases detection range but may reduce speed.
- `min_neighbors`: Determines how many neighbors each candidate rectangle should have to retain it. Higher values result in fewer detections but with higher quality.
- `detection_interval`: The interval between consecutive detections. Increasing this value can improve performance on slower systems.

### 5. Speed/Accuracy Optimization
- The system is optimized for speed. If detection is unstable:
  - Improve lighting conditions
  - Adjust parameters in `config.ini` for your setup
  - Ensure stable camera positioning

For more details, see `config.ini` and the README.

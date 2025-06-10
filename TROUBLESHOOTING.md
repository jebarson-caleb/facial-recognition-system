# 🔧 Troubleshooting Guide - Facial Recognition & Emotion Detection

## Common Issues

### 1. Camera Won't Open

**Problem**: Error message about camera not opening.

**Solutions**:
- Close all other applications using the camera (Skype, Teams, etc.)
- Check camera privacy settings in Windows
- Try unplugging and reconnecting USB camera
- Restart the application
- Try a different camera index (modify code to use cv2.VideoCapture(1) instead of 0)
- Make sure no other application is using the camera.
- Try a different USB port or restart your computer.

### 2. No Face Detected

**Problem**: Your face is not being detected at all.

**Solutions**:
- Check lighting - ensure your face is well-lit
- Position yourself 2-3 feet from the camera
- Look directly at the camera
- Close other applications using the camera
- Try pressing `c` to recalibrate camera
- Lower `min_neighbors` in config.ini for more sensitive detection
- Ensure your face is well-lit and clearly visible.
- Adjust your position (2-3 feet from camera).
- Check detection parameters in `config.ini`.

### 3. Emotion Not Detected or Always Neutral

**Problem**: Emotion predictions seem random or inaccurate.

**Solutions**:
- Make sure `models/emotion_model.h5` exists and is a valid pre-trained model.
- Check that TensorFlow is installed and up to date.
- Ensure your face is large and clear in the frame.

### 4. Detection is Slow or Laggy

**Problem**: The system responds very slowly to movement.

**Solutions**:
- Press `f` key to speed up detection
- Decrease `detection_interval` in config.ini
- Reduce `smoothing_factor` for faster response
- Ensure your computer meets performance requirements
- Close other applications using the camera or heavy system resources.
- Lower the resolution in `src/main.py` if needed.

### 5. Other Errors

**Problem**: Various error messages in the terminal.

**Solutions**:
- Check the terminal for error messages.
- Ensure all dependencies in `requirements.txt` are installed.

For more help, see the README or open an issue.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Slow down detection (increase stability) |
| `f` | Speed up detection (faster response) |
| `r` | Reset face tracking |
| `c` | Recalibrate camera |
| `Space` | Pause/unpause (if implemented) |

## Optimal Setup

1. **Lighting**: Bright, even lighting on your face
2. **Distance**: 2-3 feet from camera
3. **Position**: Face camera directly
4. **Background**: Plain, uncluttered background
5. **Movement**: Slow, deliberate movements
6. **Environment**: Quiet room without distractions

## Performance Monitoring

The system displays:
- **FPS**: Frames per second (aim for 15-30)
- **Faces Detected**: Number of faces found
- **Detection Rate**: How often detection runs
- **Frame Count**: Total frames processed
- **Confidence**: Detection confidence percentage

## Advanced Configuration

Edit `config.ini` to fine-tune:
- Detection sensitivity
- Tracking parameters
- Camera settings
- Display options

## Getting Help

If issues persist:
1. Check system requirements
2. Update camera drivers
3. Verify Python package versions
4. Test with different lighting conditions
5. Try with a different camera if available

## System Requirements

- Python 3.7+
- OpenCV 4.x
- Webcam or USB camera
- Sufficient CPU for real-time processing
- Good lighting conditions
- 2GB+ RAM recommended

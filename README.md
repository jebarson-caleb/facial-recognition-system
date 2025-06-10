# Facial Recognition & Emotion Detection System

This project provides fast, accurate, real-time face and emotion detection using your webcam. It uses a robust, pre-trained deep learning model for emotion recognition and MTCNN (or OpenCV) for face detection, with minimal dependencies and maximum speed.

## Project Structure

```
facial-recognition-system/
├── src/
│   ├── main.py                # Main entry point for live camera detection
│   ├── face_detector_advanced.py # Advanced face detector (MTCNN/haar)
│   └── emotion_detector.py    # Loads pre-trained model, predicts emotion
├── models/
│   └── emotion_model.h5       # Pre-trained Keras emotion model (mini_XCEPTION, FER2013)
├── data/
│   └── sample_images_dir/     # Sample images for testing (optional)
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── PERFORMANCE_GUIDE.md       # Tips for best results
├── TROUBLESHOOTING.md         # Common issues and solutions
├── config.ini                 # Detection parameters (optional)
```

## Setup Instructions

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Download the pre-trained model:**
   - Place `emotion_model.h5` (mini_XCEPTION, FER2013) in the `models/` directory. [Download from oarriaga/face_classification](https://github.com/oarriaga/face_classification)

## Usage

To run live face and emotion detection from your webcam:
```powershell
python src/main.py
```

If no camera is available, the system will try to use a sample image from `data/sample_images_dir/`.

## Features
- **Live Face Detection:** Fast, accurate detection using MTCNN (preferred) or OpenCV Haar cascades.
- **Emotion Recognition:** Uses a robust, pre-trained deep learning model for 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **Minimal UI:** No overlays or extra panels for maximum speed.
- **Minimal Dependencies:** Only essential libraries required.

## Notes
- For best results, ensure good lighting and position your face clearly in front of the camera.
- All unnecessary files, overlays, and extra scripts have been removed for a clean, production-ready system.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
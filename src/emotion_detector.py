import cv2
import numpy as np
import logging
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class AdvancedEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = None
        self._load_pretrained_model()

    def _load_pretrained_model(self):
        model_path = os.path.join('models', 'emotion_model.h5')
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, compile=False)
                print("Loaded pre-trained emotion model.")
            except Exception as e:
                print(f"Failed to load emotion model: {e}")
        else:
            print("No pre-trained emotion model found. Please download and place it in the models/ directory.")

    def predict_emotion(self, face_image, face_landmarks=None):
        if face_image is None or face_image.size == 0 or self.model is None:
            return "Unknown", 0.0
        try:
            processed_face = self._preprocess_face(face_image)
            face_input = processed_face.reshape(1, 64, 64, 1)
            predictions = self.model.predict(face_input, verbose=0)
            emotion_prob = predictions[0]
            emotion_idx = np.argmax(emotion_prob)
            emotion = self.emotions[emotion_idx]
            confidence = float(emotion_prob[emotion_idx])
            return emotion, confidence
        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return "Neutral", 0.0

    def _preprocess_face(self, face_image):
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        # Resize to (64, 64) to match model input
        face_resized = cv2.resize(gray, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        return face_normalized
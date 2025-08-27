import cv2
from face_detector_advanced import AdvancedFaceDetector
from emotion_detector import AdvancedEmotionDetector

def main():
    face_detector = AdvancedFaceDetector()
    emotion_detector = AdvancedEmotionDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Trying with static image...")
        test_static_image(face_detector, emotion_detector)
        return
    # Minimal camera settings for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # No extra warmup, no extra settings
    print("Camera opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        faces = face_detector.detect_faces(frame)
        print(f"Detected {len(faces)} faces.")
        for face in faces:
            x, y, w, h = face['bbox']
            if w < 30 or h < 30 or x < 0 or y < 0:
                continue
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            face_img = frame[y:y+h, x:x+w]
            if face_img is not None and face_img.size > 0:
                print(f"Processing face at ({x},{y},{w},{h})")
                emotion, confidence = emotion_detector.predict_emotion(face_img)
                face_detector.draw_faces(frame, [face])
                cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                print(f"Skipped empty face image at ({x},{y},{w},{h})")
        cv2.imshow('Face & Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_static_image(face_detector, emotion_detector):
    import os
    image_paths = [
        'data/sample_images_dir/test_face.jpg',
        'data/sample_images_dir/sample.jpg',
        '../data/sample_images_dir/test_face.jpg'
    ]
    image = None
    for image_path in image_paths:
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            print(f"Using image: {image_path}")
            break
    if image is None:
        import numpy as np
        image = np.ones((400, 400, 3), dtype=np.uint8) * 128
        cv2.putText(image, "No camera available", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    faces = face_detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face['bbox']
        face_img = image[y:y+h, x:x+w]
        emotion, confidence = emotion_detector.predict_emotion(face_img)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow('Static Image Facial Recognition and Emotion Detection', image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
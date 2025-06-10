import cv2
import numpy as np
from mtcnn import MTCNN
import logging

# Suppress MTCNN warnings
logging.getLogger('mtcnn').setLevel(logging.ERROR)

class AdvancedFaceDetector:
    def __init__(self):
        """Initialize advanced face detector using MTCNN (Multi-task CNN)"""
        try:
            # Initialize MTCNN detector
            self.detector = MTCNN()
            print("MTCNN Face Detector initialized successfully")
            self.mtcnn_available = True
        except Exception as e:
            print(f"MTCNN initialization failed: {e}")
            # Fallback to OpenCV Haar Cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.mtcnn_available = False
            print("Fallback to OpenCV Haar Cascades")
        
        # Tracking variables
        self.previous_faces = []
        self.frame_count = 0
        self.tracking_threshold = 50
        self.confidence_threshold = 0.9  # MTCNN confidence threshold
        
        # Face quality assessment
        self.min_face_size = 40
        self.max_face_size = 500
        
    def detect_faces(self, frame):
        """Main face detection method using MTCNN or fallback to OpenCV"""
        self.frame_count += 1
        
        if self.mtcnn_available:
            return self._detect_with_mtcnn(frame)
        else:
            return self._detect_with_opencv(frame)
    
    def _detect_with_mtcnn(self, frame):
        """Advanced face detection using MTCNN"""
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces with MTCNN
            detections = self.detector.detect_faces(rgb_frame)
            
            faces = []
            for detection in detections:
                # Extract bounding box and confidence
                x, y, width, height = detection['box']
                confidence = detection['confidence']
                
                # Filter by confidence and size
                if (confidence >= self.confidence_threshold and 
                    self.min_face_size <= width <= self.max_face_size and 
                    self.min_face_size <= height <= self.max_face_size):
                    
                    # Ensure coordinates are positive
                    x = max(0, x)
                    y = max(0, y)
                    
                    # Extract facial landmarks
                    keypoints = detection['keypoints']
                    
                    face_data = {
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'keypoints': keypoints,
                        'method': 'mtcnn',
                        'quality': self._calculate_face_quality_mtcnn(detection)
                    }
                    faces.append(face_data)
            
            # Apply temporal tracking
            tracked_faces = self._apply_tracking(faces)
            
            return tracked_faces
            
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return []
    
    def _detect_with_opencv(self, frame):
        """Fallback detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            maxSize=(self.max_face_size, self.max_face_size)
        )
        
        face_data_list = []
        for (x, y, w, h) in faces:
            face_data = {
                'bbox': (x, y, w, h),
                'confidence': 0.8,  # Default confidence for OpenCV
                'keypoints': None,
                'method': 'opencv',
                'quality': self._calculate_face_quality_opencv(frame[y:y+h, x:x+w])
            }
            face_data_list.append(face_data)
        
        return self._apply_tracking(face_data_list)
    
    def _calculate_face_quality_mtcnn(self, detection):
        """Calculate face quality based on MTCNN detection"""
        confidence = detection['confidence']
        
        # Check facial landmarks quality
        keypoints = detection['keypoints']
        landmarks_quality = 1.0
        
        if keypoints:
            # Calculate variance in keypoint positions for stability
            points = np.array(list(keypoints.values()))
            if len(points) > 0:
                # Higher variance in well-defined faces
                landmarks_quality = min(1.0, np.var(points) / 1000.0)
        
        # Combine confidence and landmarks quality
        quality = (confidence * 0.7 + landmarks_quality * 0.3)
        return min(1.0, quality)
    
    def _calculate_face_quality_opencv(self, face_region):
        """Calculate face quality for OpenCV detection"""
        try:
            if face_region.size == 0:
                return 0.5
            
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            # Contrast
            contrast_score = min(1.0, np.std(gray_face) / 64.0)
            
            # Size score
            size_score = min(1.0, face_region.shape[0] * face_region.shape[1] / 5000.0)
            
            quality = (sharpness_score * 0.4 + contrast_score * 0.4 + size_score * 0.2)
            return quality
            
        except Exception:
            return 0.5
    
    def _apply_tracking(self, faces):
        """Apply temporal tracking for stability"""
        if not faces:
            return []
        
        tracked_faces = []
        
        for face in faces:
            x, y, w, h = face['bbox']
            face_center = np.array([x + w/2, y + h/2])
            
            # Find closest previous face
            best_match = None
            min_distance = float('inf')
            
            for prev_face in self.previous_faces:
                px, py, pw, ph = prev_face['bbox']
                prev_center = np.array([px + pw/2, py + ph/2])
                distance = np.linalg.norm(face_center - prev_center)
                
                if distance < min_distance and distance < self.tracking_threshold:
                    min_distance = distance
                    best_match = prev_face
            
            if best_match:
                # Apply smoothing
                px, py, pw, ph = best_match['bbox']
                smooth_factor = 0.6  # Stronger smoothing for stability
                
                smoothed_x = int(smooth_factor * px + (1 - smooth_factor) * x)
                smoothed_y = int(smooth_factor * py + (1 - smooth_factor) * y)
                smoothed_w = int(smooth_factor * pw + (1 - smooth_factor) * w)
                smoothed_h = int(smooth_factor * ph + (1 - smooth_factor) * h)
                
                tracked_face = face.copy()
                tracked_face['bbox'] = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
                tracked_face['confidence'] = min(1.0, face['confidence'] + 0.05)  # Small tracking bonus
                tracked_faces.append(tracked_face)
            else:
                tracked_faces.append(face)
        
        # Update previous faces
        self.previous_faces = tracked_faces.copy()
        
        return tracked_faces
    
    def draw_faces(self, image, faces):
        """Draw detected faces with detailed information"""
        for i, face in enumerate(faces):
            x, y, w, h = face['bbox']
            confidence = face.get('confidence', 0.5)
            quality = face.get('quality', 0.5)
            method = face.get('method', 'unknown')
            keypoints = face.get('keypoints', None)
            
            # Color based on confidence and method
            if method == 'mtcnn':
                if confidence > 0.95:
                    color = (0, 255, 0)  # Bright green for high MTCNN confidence
                    thickness = 3
                elif confidence > 0.9:
                    color = (0, 200, 200)  # Cyan for good MTCNN confidence
                    thickness = 2
                else:
                    color = (0, 150, 255)  # Orange for lower MTCNN confidence
                    thickness = 2
            else:  # OpenCV
                color = (255, 0, 255)  # Magenta for OpenCV detection
                thickness = 2
            
            # Draw main rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw enhanced corner markers
            corner_size = min(20, w//4, h//4)
            self._draw_corner_markers(image, x, y, w, h, color, thickness, corner_size)
            
            # Draw facial landmarks if available (MTCNN)
            if keypoints:
                self._draw_landmarks(image, keypoints, color)
            
            # Info panel
            panel_height = 100
            panel_width = max(220, w)
            cv2.rectangle(image, (x, y - panel_height), (x + panel_width, y), (0, 0, 0), -1)
            cv2.rectangle(image, (x, y - panel_height), (x + panel_width, y), color, 2)
            
            # Display information
            cv2.putText(image, f'Face {i+1} ({method.upper()})', 
                       (x + 5, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, f'Confidence: {confidence:.3f}', 
                       (x + 5, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f'Quality: {quality:.3f}', 
                       (x + 5, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f'Size: {w}x{h}', 
                       (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Center point with method indicator
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(image, (center_x, center_y), 6, color, -1)
            cv2.circle(image, (center_x, center_y), 10, color, 2)
            
            # Method indicator
            if method == 'mtcnn':
                cv2.putText(image, 'MTCNN', (center_x - 25, center_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return image
    
    def _draw_corner_markers(self, image, x, y, w, h, color, thickness, size):
        """Draw enhanced corner markers"""
        # Top-left
        cv2.line(image, (x, y), (x + size, y), color, thickness + 1)
        cv2.line(image, (x, y), (x, y + size), color, thickness + 1)
        
        # Top-right
        cv2.line(image, (x + w, y), (x + w - size, y), color, thickness + 1)
        cv2.line(image, (x + w, y), (x + w, y + size), color, thickness + 1)
        
        # Bottom-left
        cv2.line(image, (x, y + h), (x + size, y + h), color, thickness + 1)
        cv2.line(image, (x, y + h), (x, y + h - size), color, thickness + 1)
        
        # Bottom-right
        cv2.line(image, (x + w, y + h), (x + w - size, y + h), color, thickness + 1)
        cv2.line(image, (x + w, y + h), (x + w, y + h - size), color, thickness + 1)
    
    def _draw_landmarks(self, image, keypoints, color):
        """Draw facial landmarks from MTCNN detection"""
        try:
            # Define landmark points
            landmarks = {
                'left_eye': keypoints['left_eye'],
                'right_eye': keypoints['right_eye'],
                'nose': keypoints['nose'],
                'mouth_left': keypoints['mouth_left'],
                'mouth_right': keypoints['mouth_right']
            }
            
            # Draw landmarks
            for name, point in landmarks.items():
                cv2.circle(image, tuple(map(int, point)), 3, color, -1)
                cv2.circle(image, tuple(map(int, point)), 5, color, 1)
                
                # Label landmarks
                cv2.putText(image, name.split('_')[0][:3], 
                           (int(point[0]) + 8, int(point[1]) - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        except Exception:
            pass  # Skip landmarks if there's an error
    
    def reset_tracking(self):
        """Reset face tracking"""
        self.previous_faces = []
        print("Face tracking reset")
    
    def adjust_sensitivity(self, increase=True):
        """Adjust detection sensitivity"""
        if self.mtcnn_available:
            if increase:
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
                print(f"MTCNN sensitivity increased (threshold: {self.confidence_threshold:.2f})")
            else:
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                print(f"MTCNN sensitivity decreased (threshold: {self.confidence_threshold:.2f})")
        else:
            print("Sensitivity adjustment only available with MTCNN")

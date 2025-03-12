import cv2
import numpy as np
import time
import pyautogui
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class EyeTrackerCursor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices in MediaPipe Face Mesh
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Define the screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize calibration data
        self.calibration_data = []
        self.calibration_points = []
        
        # Initialize models
        self.model_x = None
        self.model_y = None
        
        # Mode settings
        self.mode = "cursor"  # Default mode: cursor control
        self.blink_threshold = 0.2
        self.last_blink_time = time.time()
        self.blink_cooldown = 0.5  # Seconds
        
        # Initialize camera
        self.cap = None
        
    def start_camera(self):
        """Initialize the webcam capture"""
        self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate the eye aspect ratio to detect blinks"""
        # Compute the euclidean distances between the vertical eye landmarks
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Compute the eye aspect ratio
        ear = (v1 + v2) / (2.0 * h)
        
        return ear
    
    def extract_eye_features(self, landmarks):
        """Extract eye landmark features"""
        left_eye_landmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in self.LEFT_EYE])
        right_eye_landmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in self.RIGHT_EYE])
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye_landmarks, axis=0)
        right_eye_center = np.mean(right_eye_landmarks, axis=0)
        
        # Calculate eye aspect ratios
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks)
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks)
        
        # Return features
        features = np.concatenate([
            left_eye_center, 
            right_eye_center, 
            left_eye_landmarks.flatten(), 
            right_eye_landmarks.flatten(),
            [left_ear, right_ear]
        ])
        
        return features
    
    def calibrate(self, num_points=9):
        """Calibrate the eye tracker with user data"""
        if not self.start_camera():
            print("Failed to open camera")
            return False
        
        # Generate calibration points on screen
        x_points = [self.screen_width * 0.1, self.screen_width * 0.5, self.screen_width * 0.9]
        y_points = [self.screen_height * 0.1, self.screen_height * 0.5, self.screen_height * 0.9]
        
        for x in x_points:
            for y in y_points:
                self.calibration_points.append((int(x), int(y)))
        
        # Calibration process
        for point_idx, (x, y) in enumerate(self.calibration_points):
            print(f"Calibration point {point_idx + 1}/{num_points}: Look at position ({x}, {y})")
            
            # Move cursor to calibration point
            pyautogui.moveTo(x, y)
            
            # Wait for user to focus on the point
            countdown = 3
            while countdown > 0:
                print(f"Starting in {countdown}...")
                time.sleep(1)
                countdown -= 1
            
            # Collect data for 2 seconds
            start_time = time.time()
            point_data = []
            
            while time.time() - start_time < 2:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    features = self.extract_eye_features(landmarks)
                    point_data.append(features)
            
            # Add the average of collected data to calibration data
            if point_data:
                avg_features = np.mean(np.array(point_data), axis=0)
                self.calibration_data.append((avg_features, (x, y)))
                print(f"Calibration point {point_idx + 1} complete")
            else:
                print(f"Failed to collect data for point {point_idx + 1}")
        
        # Train the model
        if len(self.calibration_data) > 0:
            X = np.array([data[0] for data in self.calibration_data])
            y_x = np.array([data[1][0] for data in self.calibration_data])
            y_y = np.array([data[1][1] for data in self.calibration_data])
            
            # Train models for X and Y prediction
            self.model_x = RandomForestRegressor(n_estimators=100)
            self.model_y = RandomForestRegressor(n_estimators=100)
            
            self.model_x.fit(X, y_x)
            self.model_y.fit(X, y_y)
            
            print("Calibration complete and model trained")
            return True
        else:
            print("Calibration failed - not enough data")
            return False
    
    def save_models(self, filename_prefix="eye_tracker_model"):
        """Save the trained models"""
        import joblib
        
        if self.model_x and self.model_y:
            joblib.dump(self.model_x, f"{filename_prefix}_x.pkl")
            joblib.dump(self.model_y, f"{filename_prefix}_y.pkl")
            print(f"Models saved as {filename_prefix}_x.pkl and {filename_prefix}_y.pkl")
            return True
        else:
            print("No models to save. Please calibrate first.")
            return False
    
    def load_models(self, filename_prefix="eye_tracker_model"):
        """Load saved models"""
        import joblib
        
        try:
            self.model_x = joblib.load(f"{filename_prefix}_x.pkl")
            self.model_y = joblib.load(f"{filename_prefix}_y.pkl")
            print("Models loaded successfully")
            return True
        except FileNotFoundError:
            print("Model files not found")
            return False
    
    def switch_mode(self):
        """Switch between different control modes"""
        modes = ["cursor", "scroll", "click"]
        current_idx = modes.index(self.mode)
        next_idx = (current_idx + 1) % len(modes)
        self.mode = modes[next_idx]
        print(f"Mode switched to: {self.mode}")
    
    def handle_blink(self, left_ear, right_ear):
        """Handle blink detection and actions"""
        current_time = time.time()
        avg_ear = (left_ear + right_ear) / 2
        
        # Check if blink detected and cooldown has passed
        if avg_ear < self.blink_threshold and current_time - self.last_blink_time > self.blink_cooldown:
            self.last_blink_time = current_time
            
            if self.mode == "cursor":
                # Click where looking
                pyautogui.click()
                print("Click!")
            elif self.mode == "scroll":
                # Toggle scroll mode
                self.switch_mode()
            elif self.mode == "click":
                # Toggle click mode
                self.switch_mode()
    
    def handle_scroll(self, y_position):
        """Handle scroll behavior based on eye position"""
        if self.mode == "scroll":
            bottom_threshold = self.screen_height * 0.8
            top_threshold = self.screen_height * 0.2
            
            if y_position > bottom_threshold:
                # Scroll down
                pyautogui.scroll(-5)
            elif y_position < top_threshold:
                # Scroll up
                pyautogui.scroll(5)
    
    def run(self):
        """Run the eye tracker in continuous mode"""
        if not self.cap or not self.cap.isOpened():
            if not self.start_camera():
                print("Failed to open camera")
                return
        
        if not self.model_x or not self.model_y:
            print("Models not trained. Please calibrate first.")
            return
        
        print("Eye tracker running. Press 'q' to quit, 'm' to switch modes.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                
                # Extract the ear values from the features
                left_ear = features[-2]
                right_ear = features[-1]
                
                # Predict cursor position
                x_pred = self.model_x.predict([features])[0]
                y_pred = self.model_y.predict([features])[0]
                
                # Apply smoothing
                x_pred = max(0, min(x_pred, self.screen_width))
                y_pred = max(0, min(y_pred, self.screen_height))
                
                # Move cursor
                pyautogui.moveTo(x_pred, y_pred, duration=0.1)
                
                # Handle blinks for clicks
                self.handle_blink(left_ear, right_ear)
                
                # Handle scrolling
                self.handle_scroll(y_pred)
                
                # Draw prediction on frame
                cv2.putText(frame, f"Mode: {self.mode}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {(left_ear + right_ear)/2:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Eye Tracker", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.switch_mode()
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeTrackerCursor()
    
    # Ask user whether to calibrate or use saved model
    choice = input("Do you want to calibrate (c) or use saved model (s)? ")
    
    if choice.lower() == 'c':
        print("Starting calibration...")
        if tracker.calibrate():
            # Save model after calibration
            tracker.save_models()
            tracker.run()
        else:
            print("Calibration failed")
    elif choice.lower() == 's':
        if tracker.load_models():
            tracker.run()
        else:
            print("Failed to load models. Please calibrate first.")
    else:
        print("Invalid choice")
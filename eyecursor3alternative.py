import cv2
import numpy as np
import time
import pyautogui
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import platform

# TODO: Make the eyetracking much better and much more precise anbd the blinking work better 

class EyeTrackerCursor:
    def __init__(self):
        # Check if we're on macOS
        self.is_mac = platform.system() == 'Darwin'
        
        # Initialize MediaPipe with higher precision settings for macOS
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,  # Increased for Mac cameras
            min_tracking_confidence=0.6    # Increased for Mac cameras
        )
        
        # More comprehensive eye landmarks for better tracking
        # MediaPipe Face Mesh landmarks for eyes
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris landmarks for more precise tracking
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Define the screen size for macOS (handling Retina displays)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # For Mac Retina displays, adjust the scaling factor
        if self.is_mac:
            # Check for Retina display
            try:
                import AppKit
                main_screen = AppKit.NSScreen.mainScreen()
                backing_scale_factor = main_screen.backingScaleFactor()
                if backing_scale_factor > 1.0:
                    print(f"Detected Retina display with scale factor: {backing_scale_factor}")
            except ImportError:
                print("AppKit not available, assuming standard display")
        
        # Initialize calibration data
        self.calibration_data = []
        self.calibration_points = []
        
        # Initialize models and scaler for better prediction
        self.model_x = None
        self.model_y = None
        self.scaler = StandardScaler()
        
        # Enhanced mode settings with Mac-specific parameters
        self.mode = "cursor"  # Default mode: cursor control
        self.blink_threshold = 0.2  # Adjusted for Mac camera sensitivity
        self.last_blink_time = time.time()
        self.blink_cooldown = 0.05  # Increased to prevent accidental clicks
        self.long_blink_threshold = 1.0  # Seconds for long blink detection
        self.blink_start_time = None
        
        # Smoothing parameters for cursor movement
        self.smoothing_factor = 0.8  # How much to smooth movement (higher = smoother)
        self.prev_x, self.prev_y = None, None
        self.cursor_speed = 0.05  # Reduced for more controlled movement on Mac
        
        # Camera settings
        self.cap = None
        self.camera_index = 0  # Default camera
        self.frame_width = 1280  # Higher resolution for Mac cameras
        self.frame_height = 720
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.expanduser("~/myvscode/my/Buildownx/Eye/AlternativeEyeTrackerData")
        os.makedirs(self.data_dir, exist_ok=True)
        
    def start_camera(self):
        """Initialize the webcam capture with Mac-optimized settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Failed to open default camera. Trying alternative...")
            # Try an alternative camera index that might work on Mac
            self.camera_index = 1
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if self.cap.isOpened():
            # Set higher resolution for Mac cameras
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Set focus mode to auto for Mac cameras if available
            if self.is_mac:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
            # Get actual camera resolution
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera initialized at resolution: {actual_width}x{actual_height}")
            
            return True
        else:
            print("Failed to open any camera")
            return False
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate the eye aspect ratio with improved algorithm for Mac cameras"""
        # Compute the euclidean distances between the vertical eye landmarks
        # Using more points than original for better accuracy
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        v3 = np.linalg.norm(eye_landmarks[3] - eye_landmarks[7])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[8])
        
        # Compute the eye aspect ratio with additional vertical measurements
        ear = (v1 + v2 + v3) / (3.0 * h)
        
        return ear
    
    def extract_eye_features(self, landmarks):
        """Extract comprehensive eye landmark features"""
        # Extract full eye contour landmarks
        left_eye_landmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in self.LEFT_EYE])
        right_eye_landmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in self.RIGHT_EYE])
        
        # Extract iris landmarks for better tracking
        left_iris = np.array([[landmarks[i].x, landmarks[i].y] for i in self.LEFT_IRIS])
        right_iris = np.array([[landmarks[i].x, landmarks[i].y] for i in self.RIGHT_IRIS])
        
        # Calculate eye centers using iris center for better precision
        left_eye_center = np.mean(left_iris, axis=0)
        right_eye_center = np.mean(right_iris, axis=0)
        
        # Calculate eye aspect ratios using more landmarks for accuracy
        left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks[:9])  # Using first 9 points for EAR
        right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks[:9])
        
        # Calculate the distance between iris and eye corners for additional features
        left_iris_to_corner = np.linalg.norm(left_eye_center - left_eye_landmarks[0])
        right_iris_to_corner = np.linalg.norm(right_eye_center - right_eye_landmarks[0])
        
        # New 3D features for head pose estimation
        nose_tip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        chin = np.array([landmarks[199].x, landmarks[199].y, landmarks[199].z])
        left_eye_left = np.array([landmarks[self.LEFT_EYE[0]].x, landmarks[self.LEFT_EYE[0]].y, landmarks[self.LEFT_EYE[0]].z])
        right_eye_right = np.array([landmarks[self.RIGHT_EYE[8]].x, landmarks[self.RIGHT_EYE[8]].y, landmarks[self.RIGHT_EYE[8]].z])
        
        # Calculate head pose indicators
        head_pitch = np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2])
        head_yaw = np.arctan2(right_eye_right[0] - left_eye_left[0], right_eye_right[2] - left_eye_left[2])
        
        # Add relative positioning features (ratios rather than absolute positions)
        face_width = np.linalg.norm(right_eye_right - left_eye_left)
        left_iris_rel_x = (left_eye_center[0] - left_eye_landmarks[0][0]) / face_width
        right_iris_rel_x = (right_eye_center[0] - right_eye_landmarks[0][0]) / face_width
        
        # Return enhanced features
        features = np.concatenate([
            left_eye_center,
            right_eye_center,
            np.mean(left_iris, axis=0),  # Iris centers
            np.mean(right_iris, axis=0),
            [left_ear, right_ear],
            [left_iris_to_corner, right_iris_to_corner],
            # Head position features to compensate for head movement
            [landmarks[1].x, landmarks[1].y],  # Nose tip
            [landmarks[199].x, landmarks[199].y],  # Chin
            # New head pose features
            [head_pitch, head_yaw],
            [left_iris_rel_x, right_iris_rel_x],
            [face_width]
        ])
        
        return features
    
    def calibrate(self, num_points=36):
        """Enhanced calibration with more points and validation for Mac"""
        if not self.start_camera():
            print("Failed to open camera")
            return False
        
        # Generate more calibration points on screen for better accuracy
        x_points = [self.screen_width * 0.1, self.screen_width * 0.3, 
                   self.screen_width * 0.5, self.screen_width * 0.7, 
                   self.screen_width * 0.9
                   #, self.screen_width * 0.95
                   ]
        y_points = [self.screen_height * 0.1, self.screen_height * 0.3, 
                   self.screen_height * 0.5, self.screen_height * 0.7,
                   self.screen_height * 0.9
                   #, self.screen_width * 0.95
                   ]
        
        # Create a grid of calibration points
        self.calibration_points = []
        for x in x_points:
            for y in y_points:
                # Don't use all combinations to reduce calibration time
                if len(self.calibration_points) < num_points:
                    self.calibration_points.append((int(x), int(y)))
        
        print(f"Starting calibration with {len(self.calibration_points)} points...")
        print("Please follow the cursor and focus on each point when it appears.")
        time.sleep(2)  # Give user time to prepare
        
        # Calibration process with visual feedback
        for point_idx, (x, y) in enumerate(self.calibration_points):
            # Create a small calibration window with feedback
            calib_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.putText(calib_window, f"Point {point_idx+1}/{len(self.calibration_points)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(calib_window, f"Look at position ({x}, {y})", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imshow("Calibration", calib_window)
            cv2.waitKey(1)
            
            # Move cursor to calibration point
            pyautogui.moveTo(x, y)
            
            # Wait for user to focus on the point
            countdown = 3
            while countdown > 0:
                calib_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, f"Point {point_idx+1}/{len(self.calibration_points)}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Starting in {countdown}...", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.imshow("Calibration", calib_window)
                cv2.waitKey(1)
                time.sleep(1)
                countdown -= 1
            
            # Collect more data for each point (3 seconds)
            calib_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.putText(calib_window, f"Point {point_idx+1}/{len(self.calibration_points)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(calib_window, "Hold still and focus...", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imshow("Calibration", calib_window)
            cv2.waitKey(1)
            
            start_time = time.time()
            point_data = []
            
            while time.time() - start_time < 3:  # 3-second collection
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                
                if not results.multi_face_landmarks:
                    continue
                
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                point_data.append(features)
                
                # Show progress
                elapsed = time.time() - start_time
                progress = min(elapsed / 3.0, 1.0) * 100
                
                # Update calibration window with progress bar
                calib_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, f"Point {point_idx+1}/{len(self.calibration_points)}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(calib_window, "Hold still and focus...", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.rectangle(calib_window, (20, 100), (380, 130), (0, 0, 0), 2)
                cv2.rectangle(calib_window, (20, 100), (20 + int(360 * progress/100), 130), (0, 255, 0), -1)
                cv2.imshow("Calibration", calib_window)
                cv2.waitKey(1)
            
            # Filter and add the collected data
            if len(point_data) > 10:  # Ensure we have enough samples
                # Remove outliers (furthest from median)
                point_data = np.array(point_data)
                median_features = np.median(point_data, axis=0)
                distances = np.sum((point_data - median_features)**2, axis=1)
                
                # Keep the best 80% of points
                keep_indices = np.argsort(distances)[:int(len(distances) * 0.8)]
                filtered_data = point_data[keep_indices]
                
                # Add the average of filtered data to calibration data
                avg_features = np.mean(filtered_data, axis=0)
                self.calibration_data.append((avg_features, (x, y)))
                
                # Update with success message
                calib_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, f"Point {point_idx+1} complete!", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(calib_window, f"Collected {len(filtered_data)} valid samples", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.imshow("Calibration", calib_window)
                cv2.waitKey(300)  # Show success message briefly
            else:
                print(f"Failed to collect enough data for point {point_idx + 1}")
                
                # Update with failure message
                calib_window = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, f"Point {point_idx+1} failed!", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(calib_window, "Not enough valid data collected", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.imshow("Calibration", calib_window)
                cv2.waitKey(1000)  # Show failure message longer
        
        cv2.destroyWindow("Calibration")
        
        # Train the models with collected data
        if len(self.calibration_data) >= 6:  # Need minimum points for good calibration
            X = np.array([data[0] for data in self.calibration_data])
            y_x = np.array([data[1][0] for data in self.calibration_data])
            y_y = np.array([data[1][1] for data in self.calibration_data])
            
            # Scale features for better model performance
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models with more trees and max depth for better accuracy
            self.model_x = GradientBoostingRegressor(n_estimators=200, max_depth=15)
            self.model_y = GradientBoostingRegressor(n_estimators=200, max_depth=15)
            
            self.model_x.fit(X_scaled, y_x)
            self.model_y.fit(X_scaled, y_y)
            
            # Calculate and display calibration error
            x_pred = self.model_x.predict(X_scaled)
            y_pred = self.model_y.predict(X_scaled)
            
            mean_error_x = np.mean(np.abs(x_pred - y_x))
            mean_error_y = np.mean(np.abs(y_pred - y_y))
            
            print(f"Calibration complete with {len(self.calibration_data)} points")
            print(f"Average error: X={mean_error_x:.1f}px, Y={mean_error_y:.1f}px")
            
            # Display calibration quality
            quality = "Excellent" if (mean_error_x + mean_error_y) / 2 < 50 else \
                      "Good" if (mean_error_x + mean_error_y) / 2 < 100 else \
                      "Fair" if (mean_error_x + mean_error_y) / 2 < 150 else "Poor"
                      
            print(f"Calibration quality: {quality}")
            return True
        else:
            print("Calibration failed - not enough valid data points")
            return False
    
    

    def save_models(self, filename_prefix=None):
        """Save the trained models with Mac-compatible paths"""
        if filename_prefix is None:
            filename_prefix = os.path.join(self.data_dir, "eye_tracker_model")
        
        if self.model_x and self.model_y:
            joblib.dump(self.model_x, f"{filename_prefix}_x.pkl")
            joblib.dump(self.model_y, f"{filename_prefix}_y.pkl")
            joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")
            print(f"Models saved as {filename_prefix}_x.pkl and {filename_prefix}_y.pkl")
            print(f"Models stored in: {self.data_dir}")
            return True
        else:
            print("No models to save. Please calibrate first.")
            return False
    
    def load_models(self, filename_prefix=None):
        """Load saved models with Mac-compatible paths"""
        if filename_prefix is None:
            filename_prefix = os.path.join(self.data_dir, "eye_tracker_model")
        
        try:
            self.model_x = joblib.load(f"{filename_prefix}_x.pkl")
            self.model_y = joblib.load(f"{filename_prefix}_y.pkl")
            self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
            print("Models loaded successfully")
            return True
        except FileNotFoundError:
            print(f"Model files not found in {self.data_dir}")
            return False
    
    def switch_mode(self):
        """Switch between different control modes with Mac optimizations"""
        modes = ["cursor", "scroll", "click", "drag"]
        current_idx = modes.index(self.mode)
        next_idx = (current_idx + 1) % len(modes)
        self.mode = modes[next_idx]
        
        # Mac-specific adjustments for each mode
        if self.mode == "cursor":
            self.cursor_speed = 0.05  # Smooth movement
        elif self.mode == "scroll":
            self.cursor_speed = 0.02  # Precise scrolling
        elif self.mode == "click":
            self.cursor_speed = 0.04  # Balanced for clicking
        elif self.mode == "drag":
            self.cursor_speed = 0.03  # Slow for precise dragging
            
        print(f"Mode switched to: {self.mode}")
        return self.mode
    
    """
    def calibrate_blink_threshold(self):
        if not self.start_camera():
            print("Failed to open camera")
            return False
            
        print("Calibrating blink detection. Please look at the screen normally for 5 seconds...")
        
        # Create feedback window
        calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calib_window, "Blink Calibration", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calib_window, "Keep eyes open and look normally", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calib_window)
        cv2.waitKey(1)
        
        # Collect baseline EAR values
        baseline_ears = []
        blink_ears = []
        
        # Collect normal EAR values for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                
                # Get ear values from features based on their position in the array
                left_ear = features[-6]  # Adjust index if needed after changing extract_eye_features
                right_ear = features[-5]  # Adjust index if needed after changing extract_eye_features
                avg_ear = (left_ear + right_ear) / 2
                baseline_ears.append(avg_ear)
                
                # Show progress bar
                progress = min((time.time() - start_time) / 5.0, 1.0) * 100
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Blink Calibration", (150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, "Keep eyes open and look normally", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Current EAR: {avg_ear:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
                cv2.rectangle(calib_window, (50, 150), (450, 180), (0, 0, 0), 2)
                cv2.rectangle(calib_window, (50, 150), (50 + int(400 * progress/100), 180), (0, 255, 0), -1)
                cv2.imshow("Calibration", calib_window)
                cv2.waitKey(1)
        
        # Ask user to blink
        calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calib_window, "Blink Calibration", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calib_window, "Please blink normally 5 times", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calib_window, "Press SPACE when ready", 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calib_window)
        
        # Wait for user to be ready
        while True:
            if cv2.waitKey(1) & 0xFF == 32:  # SPACE key
                break
        
        # Collect blink EAR values
        blink_count = 0
        min_ear_values = []
        ear_window = []
        in_blink = False
        last_blink_time = time.time()
        
        calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calib_window, "Blink Calibration", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calib_window, f"Blink count: {blink_count}/5", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calib_window)
        
        # Continue until we have 5 blinks
        while blink_count < 5:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                
                # Get ear values from features
                left_ear = features[-6]  # Adjust index if needed
                right_ear = features[-5]  # Adjust index if needed
                avg_ear = (left_ear + right_ear) / 2
                
                # Add to rolling window for blink detection
                ear_window.append(avg_ear)
                if len(ear_window) > 5:
                    ear_window.pop(0)
                    
                # Update display
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Blink Calibration", (150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Blink count: {blink_count}/5", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Current EAR: {avg_ear:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
                cv2.imshow("Calibration", calib_window)
                
                # Detect blink - find local minimum in EAR
                if not in_blink and len(ear_window) >= 3:
                    # If there's a significant drop in EAR
                    if avg_ear < min(ear_window[:-1]) * 0.9:
                        in_blink = True
                        blink_start_time = time.time()
                
                # If in blink state, find minimum EAR value
                if in_blink:
                    # Store minimum EAR during blink
                    if avg_ear < min(ear_window):
                        min_ear = avg_ear
                    
                    # If EAR increases again or timeout, end of blink
                    if (avg_ear > min(ear_window) * 1.1) or (time.time() - blink_start_time > 1.0):
                        if time.time() - last_blink_time > 0.5:  # Avoid double counting
                            blink_count += 1
                            min_ear_values.append(min(ear_window))
                            last_blink_time = time.time()
                            
                            # Update display
                            calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                            cv2.putText(calib_window, "Blink Calibration", (150, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            cv2.putText(calib_window, f"Blink count: {blink_count}/5", 
                                    (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                            cv2.putText(calib_window, f"Blink detected! EAR: {min(ear_window):.4f}", 
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                            cv2.imshow("Calibration", calib_window)
                            cv2.waitKey(1)
                            time.sleep(0.5)  # Pause briefly to show detection
                        
                        in_blink = False
                
                cv2.waitKey(1)
        
        # Calculate personalized threshold
        if len(baseline_ears) > 10 and len(min_ear_values) > 3:
            # Filter out outliers from baseline
            baseline_ears.sort()
            filtered_baseline = baseline_ears[int(len(baseline_ears)*0.1):int(len(baseline_ears)*0.9)]
            avg_baseline = np.mean(filtered_baseline)
            
            # Get average of minimum blink EAR values
            avg_blink_min = np.mean(min_ear_values)
            
            # Set threshold halfway between baseline and min blink value
            self.blink_threshold = (avg_baseline + avg_blink_min) / 2
            
            # Check if values are very close (as in your case)
            if abs(avg_baseline - avg_blink_min) < 0.01:
                # Switch to relative mode
                self.blink_relative_mode = True
                self.blink_baseline = avg_baseline
                # Calculate relative threshold based on observed reduction
                reduction_ratio = avg_blink_min / avg_baseline
                self.blink_relative_threshold = (1 - reduction_ratio) * 0.8  # 80% of the observed reduction
                
                print(f"Normal EAR: {avg_baseline:.4f}, Blink EAR: {avg_blink_min:.4f}")
                print(f"Using relative blink detection with {self.blink_relative_threshold:.2f} threshold")
                
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Calibration Complete", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Using relative blink detection", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Normal EAR: {avg_baseline:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Blink EAR: {avg_blink_min:.4f}", 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Reduction threshold: {self.blink_relative_threshold:.2f}", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, "Press any key to continue", 
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                # Standard threshold mode
                print(f"Personalized blink threshold set to: {self.blink_threshold:.4f}")
                print(f"Normal EAR: {avg_baseline:.4f}, Blink EAR: {avg_blink_min:.4f}")
                
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Calibration Complete", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Blink threshold: {self.blink_threshold:.4f}", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Normal EAR: {avg_baseline:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Blink EAR: {avg_blink_min:.4f}", 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, "Press any key to continue", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.imshow("Calibration", calib_window)
            cv2.waitKey(0)
            cv2.destroyWindow("Calibration")
            return True
        else:
            print("Blink calibration failed - not enough data")
            return False
    """

    def calibrate_blink_threshold(self):
        """Calibrate the blink threshold based on user manually indicating blinks"""
        if not self.start_camera():
            print("Failed to open camera")
            return False
            
        print("Manual blink calibration. You will indicate when you blink.")
        
        # Create feedback window
        calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calib_window, "Manual Blink Calibration", (120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calib_window, "First, we'll record your normal eye state", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calib_window, "Keep eyes open and look normally", 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calib_window, "Press SPACE to start recording (5 seconds)", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calib_window)
        
        # Wait for user to be ready for baseline recording
        while True:
            if cv2.waitKey(1) & 0xFF == 32:  # SPACE key
                break
        
        # Collect baseline EAR values (eyes open)
        baseline_ears = []
        
        # Collect normal EAR values for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                
                # Get ear values from features based on their position in the array
                left_ear = features[-6]  # Adjust index if needed
                right_ear = features[-5]  # Adjust index if needed
                avg_ear = (left_ear + right_ear) / 2
                baseline_ears.append(avg_ear)
                
                # Show progress bar
                progress = min((time.time() - start_time) / 5.0, 1.0) * 100
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Recording Baseline", (150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, "Keep eyes open normally", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Current EAR: {avg_ear:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
                cv2.rectangle(calib_window, (50, 150), (450, 180), (0, 0, 0), 2)
                cv2.rectangle(calib_window, (50, 150), (50 + int(400 * progress/100), 180), (0, 255, 0), -1)
                cv2.imshow("Calibration", calib_window)
                cv2.waitKey(1)
        
        # Prepare for blink recording
        calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calib_window, "Blink Recording", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calib_window, "Now we'll record your blinks", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calib_window, "HOLD SPACE when blinking, RELEASE when done", 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calib_window, "We need 5 blinks - Press any key to start", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calib_window)
        cv2.waitKey(0)
        
        # Collect blink EAR values
        blink_count = 0
        blink_ear_values = []
        current_blink_ears = []
        
        while blink_count < 5:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                
                # Get ear values
                left_ear = features[-6]  # Adjust index if needed
                right_ear = features[-5]  # Adjust index if needed
                avg_ear = (left_ear + right_ear) / 2
                
                # Update display
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Blink Recording", (150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Blink count: {blink_count}/5", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Current EAR: {avg_ear:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
                
                # Check if SPACE is pressed (user is blinking)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE key pressed (user indicates they're blinking)
                    cv2.putText(calib_window, "BLINKING DETECTED!", 
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    current_blink_ears.append(avg_ear)
                elif len(current_blink_ears) > 0:  # SPACE released after blink
                    # A blink sequence has ended
                    blink_count += 1
                    # Store the minimum EAR value during this blink
                    min_blink_ear = min(current_blink_ears)
                    blink_ear_values.append(min_blink_ear)
                    current_blink_ears = []  # Reset for next blink
                    
                    # Show confirmation
                    calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                    cv2.putText(calib_window, "Blink Recorded!", (150, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.putText(calib_window, f"Blink {blink_count}/5 captured", 
                            (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    cv2.putText(calib_window, f"Blink EAR: {min_blink_ear:.4f}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.imshow("Calibration", calib_window)
                    cv2.waitKey(500)  # Brief pause to show confirmation
                    
                # Exit if ESC pressed
                if key == 27:  # ESC key
                    cv2.destroyWindow("Calibration")
                    return False
                    
                cv2.imshow("Calibration", calib_window)
        
        # Calculate personalized threshold
        if len(baseline_ears) > 10 and len(blink_ear_values) >= 3:
            # Filter out outliers from baseline
            baseline_ears.sort()
            filtered_baseline = baseline_ears[int(len(baseline_ears)*0.1):int(len(baseline_ears)*0.9)]
            avg_baseline = np.mean(filtered_baseline)
            
            # Get average of minimum blink EAR values
            avg_blink_min = np.mean(blink_ear_values)
            
            # Set threshold halfway between baseline and min blink value
            self.blink_threshold = (avg_baseline + avg_blink_min) / 2
            
            # Check if values are very close
            if abs(avg_baseline - avg_blink_min) < 0.05:
                # Switch to relative mode
                self.blink_relative_mode = True
                self.blink_baseline = avg_baseline
                # Calculate relative threshold based on observed reduction
                reduction_ratio = avg_blink_min / avg_baseline
                self.blink_relative_threshold = (1 - reduction_ratio) * 0.8  # 80% of the observed reduction
                
                print(f"Normal EAR: {avg_baseline:.4f}, Blink EAR: {avg_blink_min:.4f}")
                print(f"Using relative blink detection with {self.blink_relative_threshold:.2f} threshold")
                
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Calibration Complete", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Using relative blink detection", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Normal EAR: {avg_baseline:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Blink EAR: {avg_blink_min:.4f}", 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Reduction threshold: {self.blink_relative_threshold:.2f}", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, "Press any key to continue", 
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                # Standard threshold mode
                print(f"Personalized blink threshold set to: {self.blink_threshold:.4f}")
                print(f"Normal EAR: {avg_baseline:.4f}, Blink EAR: {avg_blink_min:.4f}")
                
                calib_window = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calib_window, "Calibration Complete", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calib_window, f"Blink threshold: {self.blink_threshold:.4f}", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Normal EAR: {avg_baseline:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, f"Blink EAR: {avg_blink_min:.4f}", 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calib_window, "Press any key to continue", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            cv2.imshow("Calibration", calib_window)
            cv2.waitKey(0)
            cv2.destroyWindow("Calibration")
            return True
        else:
            print("Blink calibration failed - not enough data")
            return False



    def handle_blink(self, left_ear, right_ear):
        """Handle blink detection and actions with improved timing for Mac"""
        current_time = time.time()
        avg_ear = (left_ear + right_ear) / 2
        print(f"Current EAR: {avg_ear:.2f}, Threshold: {self.blink_threshold}")
        
        # Determine if blink based on threshold method
        # For users with very low EAR values like 0.01-0.02
        if hasattr(self, 'blink_relative_mode') and self.blink_relative_mode:
            # Get baseline from recent history (last 30 frames, excluding potential blinks)
            if not hasattr(self, 'ear_history'):
                self.ear_history = []
            self.ear_history.append(avg_ear)
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)
            
            # Calculate dynamic baseline (upper percentile to avoid including blinks)
            if len(self.ear_history) >= 10:
                baseline = np.percentile(self.ear_history, 80)
                # Consider it a blink if below percentage of baseline
                is_blink = avg_ear < (baseline * (1 - self.blink_relative_threshold))
            else:
                is_blink = False
        else:
            # Traditional fixed threshold approach
            is_blink = avg_ear < self.blink_threshold
        
        # Start tracking blink if detected
        if is_blink and self.blink_start_time is None:
            self.blink_start_time = current_time
            return None  # No action yet
        
        # If we were in a blink and eyes are now open
        elif not is_blink and self.blink_start_time is not None:
            blink_duration = current_time - self.blink_start_time
            self.blink_start_time = None
            
            # Only process if cooldown has passed
            if current_time - self.last_blink_time > self.blink_cooldown:
                self.last_blink_time = current_time
                
                # Different actions based on blink duration
                if blink_duration < 0.3:  # Short blink
                    if self.mode == "cursor" or self.mode == "click":
                        # Left click
                        pyautogui.click()
                        print("Click!")
                        return "click"
                    elif self.mode == "drag":
                        # Toggle drag state
                        pyautogui.mouseDown() if not hasattr(self, 'dragging') or not self.dragging else pyautogui.mouseUp()
                        self.dragging = not getattr(self, 'dragging', False)
                        return "drag_toggle"
                elif 0.3 <= blink_duration < self.long_blink_threshold:  # Medium blink
                    if self.mode == "cursor" or self.mode == "click":
                        # Right click for Mac
                        pyautogui.rightClick()
                        return "right_click"
                else:  # Long blink
                    # Switch mode on long blink
                    new_mode = self.switch_mode()
                    return f"mode_switch_{new_mode}"
        
        return None
    
    def handle_scroll(self, y_position):
        """Handle scroll behavior with improved sensitivity for Mac"""
        if self.mode == "scroll":
            center_region = 0.3  # Dead zone in the middle (10% of screen)
            scroll_speed_factor = 0.5  # Mac-specific scroll speed adjustment
            
            # Calculate normalized position (0 to 1) with center region removed
            normalized_y = y_position / self.screen_height
            
            if normalized_y < 0.5 - center_region/2:
                # Scroll up with variable speed based on distance from center
                distance_from_center = (0.5 - center_region/2) - normalized_y
                scroll_amount = int(20 * distance_from_center * scroll_speed_factor)
                pyautogui.scroll(scroll_amount)
                return scroll_amount
            elif normalized_y > 0.5 + center_region/2:
                # Scroll down with variable speed based on distance from center
                distance_from_center = normalized_y - (0.5 + center_region/2)
                scroll_amount = int(-20 * distance_from_center * scroll_speed_factor)
                pyautogui.scroll(scroll_amount)
                return scroll_amount
                
        return 0
    
    def smooth_movement(self, x_pred, y_pred):
        """Position-aware adaptive smoothing"""
        # Initialize previous position if first movement
        if self.prev_x is None:
            self.prev_x, self.prev_y = x_pred, y_pred
            return x_pred, y_pred
        
        # Calculate distance from screen edges
        edge_distance_x = min(x_pred, self.screen_width - x_pred) / (self.screen_width / 2)
        edge_distance_y = min(y_pred, self.screen_height - y_pred) / (self.screen_height / 2)
        edge_distance = min(edge_distance_x, edge_distance_y)
        
        # Reduce smoothing near edges for better precision
        adjusted_smoothing = self.smoothing_factor * edge_distance
        
        # Apply adaptive smoothing (less smoothing near edges)
        smooth_x = self.prev_x * adjusted_smoothing + x_pred * (1 - adjusted_smoothing)
        smooth_y = self.prev_y * adjusted_smoothing + y_pred * (1 - adjusted_smoothing)
        
        # Update previous positions
        self.prev_x, self.prev_y = smooth_x, smooth_y
        
        return smooth_x, smooth_y
    def run(self):
        """Run the eye tracker with Mac optimizations"""
        if not self.cap or not self.cap.isOpened():
            if not self.start_camera():
                print("Failed to open camera")
                return
        
        if not self.model_x or not self.model_y:
            print("Models not trained. Please calibrate first.")
            return
        
        print(f"Eye tracker running on {'macOS' if self.is_mac else 'other OS'}.")
        print("Controls: Press 'q' to quit, 'm' to switch modes, 'c' to recalibrate, 'f' for full calibration")
        
        # Mac-specific settings
        if self.is_mac:
            pyautogui.FAILSAFE = False  # Disable corner failsafe for Mac
            
        # Performance tracking
        frame_times = []
        last_fps_update = time.time()
        
        while True:
            frame_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)  # Small delay before retry
                continue
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            # Display mode and debug info
            cv2.putText(frame, f"Mode: {self.mode}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_eye_features(landmarks)
                
                # Extract the ear values from the features
                left_ear = features[-6]  # Position depends on features we extract
                right_ear = features[-5]
                cv2.putText(frame, f"EAR: {(left_ear + right_ear)/2:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Scale features before prediction
                features_scaled = self.scaler.transform([features])[0].reshape(1, -1)
                
                # Predict cursor position
                x_pred = self.model_x.predict(features_scaled)[0]
                y_pred = self.model_y.predict(features_scaled)[0]
                
                # Apply smoothing for better experience
                x_smooth, y_smooth = self.smooth_movement(x_pred, y_pred)
                
                # Apply bounds
                x_smooth = max(0, min(x_smooth, self.screen_width-1))
                y_smooth = max(0, min(y_smooth, self.screen_height-1))
                
                # Move cursor with smoother duration
                pyautogui.moveTo(x_smooth, y_smooth, duration=self.cursor_speed)
                
                # Handle blinks for clicks
                blink_action = self.handle_blink(left_ear, right_ear)
                if blink_action:
                    cv2.putText(frame, f"Action: {blink_action}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Handle scrolling
                scroll_amount = self.handle_scroll(y_smooth)
                if scroll_amount != 0:
                    cv2.putText(frame, f"Scroll: {scroll_amount}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw prediction point for debug
                debug_x = int(x_smooth * frame.shape[1] / self.screen_width)
                debug_y = int(y_smooth * frame.shape[0] / self.screen_height)
                cv2.circle(frame, (debug_x, debug_y), 5, (0, 0, 255), -1)
            
            # Calculate and display FPS
            frame_times.append(time.time() - frame_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            if time.time() - last_fps_update > 1.0:  # Update FPS display once per second
                fps = 1.0 / (sum(frame_times) / len(frame_times))
                last_fps_update = time.time()          
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Eye Tracker", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.switch_mode()
            elif key == ord('c'):
                # Quick recalibration
                cv2.destroyAllWindows()
                self.calibrate(9)  # Use fewer points for quick recalibration
            # Inside the run() method where key presses are handled
            elif key == ord('f'):  # 'f' for full calibration
                print("Starting full calibration from scratch...")
                cv2.destroyAllWindows()
                self.calibration_data = []  # Clear any existing calibration data
                self.calibration_points = []  # Clear existing points
                if self.calibrate(16):  # Use all 16 points for a complete calibration
                    self.save_models()  # Save the new calibration
                    print("Full calibration complete. Press any key to continue.")
                    cv2.waitKey(0)
                
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # If we were in drag mode, ensure mouse is released
        if hasattr(self, 'dragging') and self.dragging:
            pyautogui.mouseUp()
            
        print("Eye tracker stopped")

if __name__ == "__main__":
    tracker = EyeTrackerCursor()
    
    # Create a simple startup GUI
    startup_window = np.ones((300, 500, 3), dtype=np.uint8) * 240
    cv2.putText(startup_window, "MacOS Eye Tracker", (120, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(startup_window, "Choose an option:", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(startup_window, "c - Calibrate (recommended for first use)", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(startup_window, "l - Load saved calibration", (50, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(startup_window, "q - Quit", (50, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imshow("Eye Tracker Setup", startup_window)
    
    choice = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    
    if choice == ord('c'):
        print("Starting calibration...")
        if tracker.calibrate():
            # Save model after calibration
            tracker.calibrate_blink_threshold()
            tracker.save_models()
            tracker.run()
        else:
            print("Calibration failed")
    elif choice == ord('l'):
        if tracker.load_models():
            tracker.run()
        else:
            print("Failed to load models. Please calibrate first.")
            # Offer to calibrate if loading fails
            retry = input("Would you like to calibrate now? (y/n): ")
            if retry.lower() == 'y':
                if tracker.calibrate():
                    tracker.save_models()
                    tracker.run()
    
    else:
        print("Exiting program")
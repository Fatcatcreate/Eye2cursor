import cv2
import numpy as np
import time
import pyautogui
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import platform

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
        self.smoothing_factor = 0.5  # How much to smooth movement (higher = smoother)
        self.prev_x, self.prev_y = None, None
        self.cursor_speed = 0.05  # Reduced for more controlled movement on Mac
        
        # Camera settings
        self.cap = None
        self.camera_index = 0  # Default camera
        self.frame_width = 1280  # Higher resolution for Mac cameras
        self.frame_height = 720
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.expanduser("~/myvscode/my/Buildownx/Eye/EyeTrackerData")
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
            [landmarks[199].x, landmarks[199].y]  # Chin
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
                   self.screen_width * 0.9, self.screen_width * 0.95]
        y_points = [self.screen_height * 0.1, self.screen_height * 0.3, 
                   self.screen_height * 0.5, self.screen_height * 0.7,
                   self.screen_height * 0.9, self.screen_width * 0.95]
        
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
            self.model_x = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1)
            self.model_y = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1)
            
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
    
    def handle_blink(self, left_ear, right_ear):
        """Handle blink detection and actions with improved timing for Mac"""
        current_time = time.time()
        avg_ear = (left_ear + right_ear) / 2
        print(f"Current EAR: {avg_ear:.2f}, Threshold: {self.blink_threshold}")
        


        # Start tracking blink if detected
        if avg_ear < self.blink_threshold and self.blink_start_time is None:
            self.blink_start_time = current_time
            return None  # No action yet
            
        # If we were in a blink and eyes are now open
        elif avg_ear >= self.blink_threshold and self.blink_start_time is not None:
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
        """Apply smoothing to cursor movement for better Mac experience"""
        # Initialize previous position if first movement
        if self.prev_x is None:
            self.prev_x, self.prev_y = x_pred, y_pred
            return x_pred, y_pred
            
        # Apply exponential smoothing
        smooth_x = self.prev_x * self.smoothing_factor + x_pred * (1 - self.smoothing_factor)
        smooth_y = self.prev_y * self.smoothing_factor + y_pred * (1 - self.smoothing_factor)
        
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
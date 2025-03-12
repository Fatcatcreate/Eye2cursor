import cv2
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import messagebox
import time
import os
import pickle
from sklearn.svm import SVR
import threading
import dlib
from collections import deque


# macOS-specific setup
pyautogui.FAILSAFE = False  # Disable fail-safe for smoother operation
# Set a more reasonable duration for cursor movement on macOS
MOVE_DURATION = 0.05

class EyeTracker:
    def __init__(self):

        # macOS-specific setup for OpenCV
        cv2.setNumThreads(1)  # Reduce threading issues
        
        # Create named windows before using them to avoid crashes
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)

        # Configuration
        self.calibration_points = 81  # Number of calibration points (9x9 grid)
        self.calibration_duration = 2  # Seconds to look at each point
        self.blink_threshold = 0.2  # Ratio of eye height to width
        self.blink_duration = 0.3  # Seconds eye must be closed to count as a blink
        self.mode = "move"  # Default mode: move, scroll, or advanced
        
        # Initialize webcam
        # On macOS, sometimes we need to try multiple indices or specify a backend
        self.cap = None
        for i in range(3):  # Try first three camera indices
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
        
        if self.cap is None or not self.cap.isOpened():
            raise Exception("Could not open webcam. Make sure your webcam is connected and that macOS has permission to access it.")
        
        # Get screen dimensions - macOS uses Retina scaling which pyautogui accounts for
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Buffer for blink detection
        self.blink_history = deque(maxlen=10)
        self.last_blink_time = time.time()
        
        # For cursor smoothing
        self.gaze_history = deque(maxlen=10)
        
        # Initialize facial landmark detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Load or download the facial landmark predictor
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Facial landmark predictor not found. Please download from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place in the same directory as this script.")
            exit(1)
        
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Models for gaze prediction
        self.model_x = None
        self.model_y = None
        
        # For calibration
        self.calibration_data = []
        self.calibration_features = []
        
        # For mode switching
        self.zone_threshold = 100  # Pixels from screen edge for scroll zones
        
        # Initialize UI
        self.root = None
        self.canvas = None
        self.info_label = None
        
        # Mac-specific: store click time to prevent double-click
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds
        
        # Mac-specific: scroll speed adjustment
        self.scroll_speed_vertical = 2
        
    def extract_eye_features(self, frame):
        """Extract features from the eyes for gaze estimation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces:
            return None, frame, 0
        
        # Use the first face detected
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Extract eye landmarks (36-47 in dlib's 68 point model)
        left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        # Draw eye contours
        cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 1)
        
        # Calculate eye aspect ratio (EAR) for blink detection
        def eye_aspect_ratio(eye_points):
            # Vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0
        
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.blink_history.append(avg_ear)
        
        # Extract features for gaze estimation
        features = []
        
        # Normalized positions of eye corners and centers
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        
        # Normalize by face size
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()
        face_center = np.array([face.left() + face_width/2, face.top() + face_height/2])
        
        # Add normalized eye positions to features
        for eye_points in [left_eye_points, right_eye_points]:
            for point in eye_points:
                normalized_point = (point - face_center) / np.array([face_width, face_height])
                features.extend(normalized_point)
        
        # Add head pose estimation features (simple approximation)
        features.extend([(right_eye_center - left_eye_center) / face_width])
        
        # Add pupil detection features
        for eye_points in [left_eye_points, right_eye_points]:
            eye_region = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(eye_region, [eye_points], 255)
            eye = cv2.bitwise_and(gray, gray, mask=eye_region)
            
            # Find the darkest point in the eye region (approximate pupil)
            eye_copy = eye.copy()
            eye_copy[eye_copy == 0] = 255
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(eye_copy)
            
            # Draw pupil
            if minLoc != (0, 0):
                cv2.circle(frame, minLoc, 2, (0, 0, 255), -1)
                # Normalize pupil position relative to eye corners
                eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
                if eye_width > 0:
                    normalized_pupil_x = (minLoc[0] - eye_points[0][0]) / eye_width
                    normalized_pupil_y = (minLoc[1] - eye_points[0][1]) / eye_width
                    features.extend([normalized_pupil_x, normalized_pupil_y])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])  # Placeholder if pupil detection fails
        
        return features, frame, avg_ear
    
    def start_calibration(self):
        """Start the calibration process with UI"""
        self.root = tk.Tk()
        self.root.title("Eye Tracker Calibration")
        
        # On macOS, use attributes that work better with macOS window manager
        self.root.attributes('-fullscreen', True)
        
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.info_label = tk.Label(
            self.root, 
            text="Calibration will start in 5 seconds.\nLook at the red dot when it appears.",
            font=("SF Pro", 24),  # macOS system font
            fg="white",
            bg="black"
        )
        self.info_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.root.after(5000, self.run_calibration)
        self.root.mainloop()
    
    def start_calibration(self):
        """Start the calibration process with UI"""
        self.root = tk.Tk()
        self.root.title("Eye Tracker Calibration")
        
        # On macOS, use attributes that work better with macOS window manager
        self.root.attributes('-fullscreen', True)
        
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.info_label = tk.Label(
            self.root, 
            text="Calibration will start in 5 seconds.\nLook at the red dot when it appears.",
            font=("SF Pro", 24),  # macOS system font
            fg="white",
            bg="black"
        )
        self.info_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Completely avoid showing OpenCV windows during Tkinter calibration
        self.disable_opencv_windows = True
        
        self.root.after(5000, self.run_calibration)
        self.root.mainloop()
        
    def run_calibration(self):
        """Run the calibration sequence"""
        self.info_label.destroy()
        self.calibration_data = []
        self.calibration_features = []
        
        # Create a 3x3 grid of calibration points
        width, height = self.root.winfo_width(), self.root.winfo_height()
        points = []
        for y in [height * 0.1, height * 0.5, height * 0.9]:
            for x in [width * 0.1, width * 0.5, width * 0.9]:
                points.append((int(x), int(y)))
        
        def show_point(index):
            if index >= len(points):
                self.finish_calibration()
                return
            
            self.canvas.delete("all")
            x, y = points[index]
            
            # Draw calibration point
            self.canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10, 
                fill="red", outline="white", width=2
            )
            
            # Update UI and process frames
            self.root.update()
            
            # Collect data for this point
            start_time = time.time()
            point_features = []
            """
            def collect_data():
                nonlocal point_features
                while time.time() - start_time < self.calibration_duration:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    
                    features, processed_frame, _ = self.extract_eye_features(frame)
                    if features:
                        point_features.append(features)
                    
                    # Show webcam feed in small window
                    
                            # Add error handling for the display
                    try:
                        resized_frame = cv2.resize(processed_frame, (320, 240))
                        cv2.imshow("Calibration", resized_frame)
                        cv2.waitKey(1)  # Ensure waitKey is called after imshow
                    except Exception as e:
                        print(f"Display warning (non-critical): {e}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.root.destroy()
                        return
                
                # Average the features for this point
                if point_features:
                    avg_features = np.mean(point_features, axis=0)
                    self.calibration_features.append(avg_features)
                    self.calibration_data.append((x, y))
                
                # Move to next point
                self.root.after(100, lambda: show_point(index + 1))
                """
            
            
            # Start collecting data in a separate thread
            threading.Thread(target=collect_data, daemon=True).start()
        
        # Start the calibration sequence
        show_point(0)
    
    def finish_calibration(self):
        """Train the gaze estimation model using calibration data"""
        cv2.destroyAllWindows()
        
        if len(self.calibration_features) < 5:
            messagebox.showerror(
                "Calibration Failed", 
                "Not enough calibration data collected. Please try again."
            )
            self.root.destroy()
            return
        
        # Train models to predict screen coordinates from eye features
        X = np.array(self.calibration_features)
        y_x = np.array([point[0] for point in self.calibration_data])
        y_y = np.array([point[1] for point in self.calibration_data])
        
        self.model_x = SVR(kernel='rbf')
        self.model_y = SVR(kernel='rbf')
        
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)
        
        messagebox.showinfo(
            "Calibration Complete", 
            "Calibration successful! The eye tracker is now ready.\n\n"
            "Controls:\n"
            "- Blink to click\n"
            "- Press 1: Move cursor mode\n"
            "- Press 2: Scroll mode\n"
            "- Press 3: Advanced mode (stare to click, blink to scroll)\n"
            "- Press 'q' to quit\n"
            "- Press 'c' to recalibrate"
        )
        
        # Save calibration data
        with open('eye_tracker_calibration.pkl', 'wb') as f:
            pickle.dump({
                'model_x': self.model_x,
                'model_y': self.model_y
            }, f)
        
        self.root.destroy()
        self.start_tracking()
    
    def load_calibration(self):
        """Load calibration data if available"""
        if os.path.exists('eye_tracker_calibration.pkl'):
            try:
                with open('eye_tracker_calibration.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.model_x = data['model_x']
                    self.model_y = data['model_y']
                return True
            except:
                return False
        return False
    
    def detect_blink(self, ear):
        """Detect blinks based on eye aspect ratio"""
        if ear < self.blink_threshold:
            # Eye is closed
            if (time.time() - self.last_blink_time) > self.blink_duration:
                self.last_blink_time = time.time()
                return True
        return False
    
    def safe_click(self):
        """Perform a click with cooldown to prevent accidental double-clicks on macOS"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            pyautogui.click()
            self.last_click_time = current_time
    
    def start_tracking(self):
        """Start the eye tracking after calibration"""
        if self.model_x is None or self.model_y is None:
            if not self.load_calibration():
                print("No calibration data found. Starting calibration...")
                self.start_calibration()
                return
        
        print("Eye tracking started!")
        print("Controls:")
        print("- Blink to click")
        print("- Press 1: Move cursor mode")
        print("- Press 2: Scroll mode")
        print("- Press 3: Advanced mode (stare to click, blink to scroll)")
        print("- Press 'q' to quit")
        print("- Press 'c' to recalibrate")
        
        # Main tracking loop
        last_mode_switch = time.time()
        mode_cooldown = 1.0  # Seconds between mode switches
        
        # Create a named window for key handling
        cv2.namedWindow("Eye Tracker", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)  # Wait a bit before trying again
                continue
            
            # Flip horizontally for more intuitive experience
            frame = cv2.flip(frame, 1)
            
            # Extract features and detect blinks
            features, processed_frame, ear = self.extract_eye_features(frame)
            
            if features is not None:
                # Predict gaze position
                pred_x = self.model_x.predict([features])[0]
                pred_y = self.model_y.predict([features])[0]
                
                # Smooth predictions with moving average
                self.gaze_history.append((pred_x, pred_y))
                smoothed_x = sum([p[0] for p in self.gaze_history]) / len(self.gaze_history)
                smoothed_y = sum([p[1] for p in self.gaze_history]) / len(self.gaze_history)
                
                # Ensure coordinates are within screen bounds
                smoothed_x = max(0, min(self.screen_width, smoothed_x))
                smoothed_y = max(0, min(self.screen_height, smoothed_y))
                
                # Process different modes
                if self.mode == "move":
                    # Move cursor - use lower duration for smoother experience on macOS
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=MOVE_DURATION)
                    
                    # Blink detection for clicking
                    if self.detect_blink(ear):
                        self.safe_click()
                
                elif self.mode == "scroll":
                    # Move cursor
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=MOVE_DURATION)
                    
                    # Scroll based on vertical position - macOS may need different values
                    if smoothed_y > self.screen_height - self.zone_threshold:
                        pyautogui.scroll(-self.scroll_speed_vertical)  # Scroll down
                    elif smoothed_y < self.zone_threshold:
                        pyautogui.scroll(self.scroll_speed_vertical)   # Scroll up
                    
                    # Blink detection for clicking
                    if self.detect_blink(ear):
                        self.safe_click()
                
                elif self.mode == "advanced":
                    # Move cursor
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=MOVE_DURATION)
                    
                    # Stare to click (dwell clicking)
                    if len(self.gaze_history) == self.gaze_history.maxlen:
                        std_x = np.std([p[0] for p in self.gaze_history])
                        std_y = np.std([p[1] for p in self.gaze_history])
                        
                        if std_x < 15 and std_y < 15:  # Stable gaze threshold
                            # Visual feedback for dwelling
                            cv2.circle(processed_frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), 
                                      20, (0, 255, 255), 2)
                            
                            if (time.time() - self.last_click_time) > 1.0:  # Dwell time
                                self.safe_click()
                    
                    # Blink detection for scrolling
                    if self.detect_blink(ear):
                        if smoothed_y > self.screen_height - self.zone_threshold:
                            pyautogui.scroll(-10)  # Scroll down more
                        elif smoothed_y < self.zone_threshold:
                            pyautogui.scroll(10)   # Scroll up more
                
                # Display gaze position on frame
                cv2.putText(
                    processed_frame, 
                    f"Gaze: ({int(smoothed_x)}, {int(smoothed_y)})", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Display mode
                cv2.putText(
                    processed_frame, 
                    f"Mode: {self.mode}", 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Display EAR value
                cv2.putText(
                    processed_frame, 
                    f"EAR: {ear:.2f}", 
                    (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
            # Display the frame
            cv2.imshow("Eye Tracker", processed_frame)
            
            # Check for key presses - on macOS, we need to ensure this is responsive
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1') and (time.time() - last_mode_switch) > mode_cooldown:
                self.mode = "move"
                last_mode_switch = time.time()
                print("Mode: Move cursor (blink to click)")
            elif key == ord('2') and (time.time() - last_mode_switch) > mode_cooldown:
                self.mode = "scroll"
                last_mode_switch = time.time()
                print("Mode: Scroll mode (look at top/bottom to scroll, blink to click)")
            elif key == ord('3') and (time.time() - last_mode_switch) > mode_cooldown:
                self.mode = "advanced"
                last_mode_switch = time.time()
                print("Mode: Advanced mode (stare to click, blink to scroll)")
            elif key == ord('c'):
                # Recalibrate
                cv2.destroyAllWindows()
                self.start_calibration()
                return
            elif key == ord('+'):
                # Increase scroll speed
                self.scroll_speed_vertical += 1
                print(f"Scroll speed: {self.scroll_speed_vertical}")
            elif key == ord('-'):
                # Decrease scroll speed
                self.scroll_speed_vertical = max(1, self.scroll_speed_vertical - 1)
                print(f"Scroll speed: {self.scroll_speed_vertical}")
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main entry point to start the eye tracker"""
        # Make sure camera permissions are set on macOS
        print("Starting Eye Tracker...")
        print("Note: On macOS, you may need to grant camera permissions in System Preferences > Security & Privacy > Camera")
        
        if self.load_calibration():
            choice = input("Calibration data found. Do you want to use it? (y/n): ")
            if choice.lower() != 'y':
                self.start_calibration()
            else:
                self.start_tracking()
        else:
            self.start_calibration()

if __name__ == "__main__":
    try:
        # On macOS, make sure we can create GUI windows
        # This prevents the "Qt internal exception" error on some macOS versions
        os.environ['QT_MAC_WANTS_LAYER'] = '1'
        
        tracker = EyeTracker()
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
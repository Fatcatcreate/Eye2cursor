import cv2
import numpy as np
import pyautogui
import time
import os
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten # type: ignore
import matplotlib.pyplot as plt

# Prevent mouse movement from causing issues
pyautogui.FAILSAFE = False

class EyeTrackingInterface:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
            
        # Set webcam properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Load pre-trained models for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            print("Warning: Haar cascades not found. Attempting to download...")
            self.download_cascades()
            # Reload cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize gaze model
        self.gaze_model = None
        
        # Calibration data
        self.calibration_points = []
        self.calibration_eye_features = []
        # Dwell click mechanism
        self.dwell_time = 1.0  # seconds
        self.dwell_start_time = None
        self.dwell_position = None
        self.dwell_threshold = 50  # pixels
        
        # Smoothing parameters
        self.smoothing_factor = 0.3
        self.last_x, self.last_y = self.screen_width // 2, self.screen_height // 2
        
        # Create data directories
        os.makedirs("eye_tracking_data", exist_ok=True)
        
        # Debug mode
        self.debug = False
        
        # Control mode (True for cursor control, False for data collection)
        self.control_mode = False

    def download_cascades(self):
        """Download cascade files if not available"""
        import urllib.request
        
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
        files = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_eye.xml"
        ]
        
        for file in files:
            url = base_url + file
            dest = os.path.join(cv2.data.haarcascades, file)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded {file}")

    def extract_eye_features(self, frame):
        """Extract features from the eye region"""
        if frame is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better eye detection in various lighting
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
        
        if len(eyes) < 2:
            return None
        
        # Sort eyes by x-coordinate to get left and right eye
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Make sure we have exactly 2 eyes
        eyes = eyes[:2]
        
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        # Extract eye regions
        left_x, left_y, left_w, left_h = left_eye
        right_x, right_y, right_w, right_h = right_eye
        
        left_eye_gray = face_gray[left_y:left_y+left_h, left_x:left_x+left_w]
        right_eye_gray = face_gray[right_y:right_y+right_h, right_x:right_x+right_w]
        
        # Resize eyes to fixed dimensions
        try:
            left_eye_resized = cv2.resize(left_eye_gray, (32, 24))
            right_eye_resized = cv2.resize(right_eye_gray, (32, 24))
        except:
            return None
        
        # Normalize pixel values
        left_eye_normalized = left_eye_resized / 255.0
        right_eye_normalized = right_eye_resized / 255.0
        
        # Draw face and eye rectangles if in debug mode
        if self.debug:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(face_color, (left_x, left_y), (left_x+left_w, left_y+left_h), (0, 255, 0), 2)
            cv2.rectangle(face_color, (right_x, right_y), (right_x+right_w, right_y+right_h), (0, 255, 0), 2)
        
        # Flatten and combine eye features for simple model
        features = np.concatenate([left_eye_normalized.flatten(), right_eye_normalized.flatten()])
        
        # Also return the original eye images for CNN model
        eye_images = np.stack([left_eye_normalized, right_eye_normalized], axis=2)  # Shape: (24, 32, 2)
        
        return features, eye_images, frame

    def run_calibration(self):
        """Run calibration process with visual feedback"""
        print("Starting calibration process...")
        
        # Clear previous calibration data
        self.calibration_points = []
        self.calibration_eye_features = []
        
        # Define calibration points - grid of 3x3
        screen_w, screen_h = self.screen_width, self.screen_height
        points = [
            (int(screen_w*0.1), int(screen_h*0.1)), 
            (int(screen_w*0.5), int(screen_h*0.1)), 
            (int(screen_w*0.9), int(screen_h*0.1)),
            (int(screen_w*0.1), int(screen_h*0.5)), 
            (int(screen_w*0.5), int(screen_h*0.5)), 
            (int(screen_w*0.9), int(screen_h*0.5)),
            (int(screen_w*0.1), int(screen_h*0.9)), 
            (int(screen_w*0.5), int(screen_h*0.9)), 
            (int(screen_w*0.9), int(screen_h*0.9))
        ]
        
        # Create full screen window for calibration
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Display instructions
        calib_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(calib_img, "Calibration Process", 
                    (screen_w//2-150, screen_h//2-50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(calib_img, "Look at each circle as it appears and hold still", 
                    (screen_w//2-250, screen_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(calib_img, "Press SPACE to start", 
                    (screen_w//2-100, screen_h//2+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", calib_img)
        
        # Wait for space key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cv2.destroyWindow("Calibration")
                return False
        
        # Process each calibration point
        for i, point in enumerate(points):
            # Create image with calibration point
            calib_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            
            # Draw point with animation (growing circle)
            for radius in range(5, 20, 5):
                calib_img_copy = calib_img.copy()
                cv2.circle(calib_img_copy, point, radius, (0, 255, 0), -1)
                cv2.putText(calib_img_copy, f"Point {i+1}/{len(points)}", 
                           (screen_w//2-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Calibration", calib_img_copy)
                cv2.waitKey(100)
            
            # Show final point
            calib_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(calib_img, point, 15, (0, 255, 0), -1)
            cv2.putText(calib_img, f"Point {i+1}/{len(points)}", 
                       (screen_w//2-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(calib_img, "Keep looking at the circle", 
                       (screen_w//2-150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Calibration", calib_img)
            
            # Collect samples for this point
            samples_to_collect = 30  # Number of samples per calibration point
            collected_samples = 0
            
            # Progress bar parameters
            progress_bar_width = int(screen_w * 0.6)
            progress_bar_height = 20
            progress_bar_x = int(screen_w * 0.2)
            progress_bar_y = screen_h - 50
            
            start_time = time.time()
            timeout = 5  # seconds
            
            while collected_samples < samples_to_collect and time.time() - start_time < timeout:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                result = self.extract_eye_features(frame)
                if result is None:
                    print("Failed to extract eye features")
                if result is not None:
                    flat_features, eye_images, _ = result
                    
                    # Add to calibration data
                    self.calibration_points.append(point)
                    self.calibration_eye_features.append(eye_images)  # Store the 2D eye images for CNN
                    collected_samples += 1
                    
                    # Update progress bar
                    progress = collected_samples / samples_to_collect
                    progress_width = int(progress_bar_width * progress)
                    
                    progress_img = calib_img.copy()
                    # Draw progress bar background
                    cv2.rectangle(progress_img, 
                                 (progress_bar_x, progress_bar_y), 
                                 (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height), 
                                 (100, 100, 100), -1)
                    # Draw progress
                    cv2.rectangle(progress_img, 
                                 (progress_bar_x, progress_bar_y), 
                                 (progress_bar_x + progress_width, progress_bar_y + progress_bar_height), 
                                 (0, 255, 0), -1)
                    
                    cv2.imshow("Calibration", progress_img)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyWindow("Calibration")
                    return False
            
            # Short delay before moving to next point
            time.sleep(0.5)
        
        # Final message
        calib_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(calib_img, "Calibration Complete!", 
                   (screen_w//2-150, screen_h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(calib_img, "Press any key to continue", 
                   (screen_w//2-150, screen_h//2+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", calib_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration")
        
        # Save calibration data
        if len(self.calibration_points) > 0:
            self.save_calibration_data()
            print(f"Calibration completed with {len(self.calibration_points)} data points")
            return True
        else:
            print("No calibration data collected")
            return False
    
    def collect_additional_data(self):
        """Collect additional training data across the screen"""
        # Create a window for data collection
        cv2.namedWindow("Data Collection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Data Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        screen_w, screen_h = self.screen_width, self.screen_height
        
        # Inform the user
        info_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(info_img, "Additional Data Collection", 
                   (screen_w//2-200, screen_h//2-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(info_img, "Follow the moving target with your eyes", 
                   (screen_w//2-250, screen_h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(info_img, "Press SPACE to start, Q to quit", 
                   (screen_w//2-200, screen_h//2+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Data Collection", info_img)
        
        # Wait for space key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cv2.destroyWindow("Data Collection")
                return False
        
        # Create a grid of points for data collection with more density than calibration
        grid_size = 5  # 5x5 grid
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(screen_w * (0.1 + 0.8 * i / (grid_size - 1)))
                y = int(screen_h * (0.1 + 0.8 * j / (grid_size - 1)))
                points.append((x, y))
        
        # Add some random points for variety
        import random
        random_points = []
        for _ in range(20):
            x = random.randint(int(screen_w * 0.1), int(screen_w * 0.9))
            y = random.randint(int(screen_h * 0.1), int(screen_h * 0.9))
            random_points.append((x, y))
        
        points.extend(random_points)
        random.shuffle(points)  # Randomize point order
        
        # Collect data for each point
        for i, point in enumerate(points):
            # Create image with data collection point
            data_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(data_img, point, 15, (0, 0, 255), -1)  # Red circle
            cv2.putText(data_img, f"Point {i+1}/{len(points)}", 
                       (screen_w//2-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Data Collection", data_img)
            
            # Collect samples
            samples_per_point = 10
            collected = 0
            
            start_time = time.time()
            timeout = 3  # seconds
            
            while collected < samples_per_point and time.time() - start_time < timeout:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                result = self.extract_eye_features(frame)
                if result is not None:
                    _, eye_images, _ = result
                    
                    # Add to calibration data (reuse the same structures)
                    self.calibration_points.append(point)
                    self.calibration_eye_features.append(eye_images)
                    collected += 1
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyWindow("Data Collection")
                    return False
            
            # Short delay before moving to next point
            time.sleep(1)
        
        # Final message
        data_img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(data_img, "Data Collection Complete!", 
                   (screen_w//2-200, screen_h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(data_img, "Press any key to continue", 
                   (screen_w//2-150, screen_h//2+50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Data Collection", data_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Data Collection")
        
        # Save data
        self.save_calibration_data()
        print(f"Additional data collection completed with total {len(self.calibration_points)} data points")
        return True

    def save_calibration_data(self):
        """Save calibration data to file"""
        # Convert lists to numpy arrays
        points = np.array(self.calibration_points)
        eye_features = np.array(self.calibration_eye_features)
        
        # Save data
        data = {
            'points': points,
            'eye_features': eye_features
        }
        
        filename = os.path.join("eye_tracking_data", f"calibration_data_{time.strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Calibration data saved to {filename}")

    def load_calibration_data(self, filename=None):
        """Load calibration data from file(s)"""
        if filename is None:
            # Load the most recent file
            files = [f for f in os.listdir("eye_tracking_data") if f.startswith("calibration_data_")]
            if not files:
                print("No calibration data found")
                return False
            
            files.sort(reverse=True)  # Most recent first
            filename = os.path.join("eye_tracking_data", files[0])
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.calibration_points = data['points'].tolist()
            self.calibration_eye_features = data['eye_features'].tolist()
            
            print(f"Loaded {len(self.calibration_points)} calibration points from {filename}")
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False

    def train_model(self):
        """Train gaze prediction model using collected data"""
        if len(self.calibration_points) < 50:
            print("Not enough calibration data to train model")
            return False
        
        print(f"Training model with {len(self.calibration_points)} data points...")
        
        # Convert lists to numpy arrays
        X = np.array(self.calibration_eye_features)  # Shape: (n_samples, 24, 32, 2)
        y = np.array(self.calibration_points)        # Shape: (n_samples, 2)
        
        # Normalize target coordinates to [0, 1] range
        y = y.astype(np.float32)
        y[:, 0] /= self.screen_width
        y[:, 1] /= self.screen_height
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and compile CNN model
        model = Sequential([
            # First convolutional layer
            Conv2D(32, (3, 3), activation='relu', input_shape=(24, 32, 2)),
            MaxPooling2D((2, 2)),
            
            # Second convolutional layer
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            # Flatten the output for the dense layers
            Flatten(),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(2)  # Output: (x, y) coordinates
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Show model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join("eye_tracking_data", "training_history.png"))
        
        # Save model
        model_path = os.path.join("eye_tracking_data", "gaze_model.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Set model for inference
        self.gaze_model = model
        
        return True
    """
    def load_model(self, model_path=None):
        print("Loading model...")
        if model_path is None:
            model_path = os.path.join("eye_tracking_data", "gaze_model.h5")
        
        try:
            self.gaze_model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    """

    def load_model(self, model_path=None):
        """Load pre-trained gaze prediction model"""
        print("Loading model...")
        if model_path is None:
            model_path = os.path.join("eye_tracking_data", "gaze_model.h5")
        
        try:
            # Define custom objects dictionary with the metrics used during training
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }
            
            # Use tf.keras.models.load_model with custom_objects
            self.gaze_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def perform_data_augmentation(self, output_filename=None):
        """Augment collected eye tracking data"""
        if len(self.calibration_points) < 50:
            print("Not enough data to perform augmentation")
            return False
        
        print("Performing data augmentation...")
        
        # Convert to numpy arrays
        X = np.array(self.calibration_eye_features)  # Shape: (n_samples, 24, 32, 2)
        y = np.array(self.calibration_points)        # Shape: (n_samples, 2)
        
        # Original data size
        original_size = X.shape[0]
        
        # List to store augmented data
        X_augmented = [X]
        y_augmented = [y]
        
        # 1. Small brightness variations
        brightness_factors = [0.8, 0.9, 1.1, 1.2]
        for factor in brightness_factors:
            X_bright = X * factor
            X_bright = np.clip(X_bright, 0, 1)  # Keep in valid range
            X_augmented.append(X_bright)
            y_augmented.append(y)
        
        # 2. Small rotations (simulate slight head tilt)
        angles = [-5, -3, 3, 5]  # degrees
        for angle in angles:
            X_rotated = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[3]):  # For each eye channel
                    # Get original image
                    img = X[i, :, :, j]
                    
                    # Get rotation matrix
                    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
                    
                    # Apply rotation
                    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    
                    # Store back
                    X_rotated[i, :, :, j] = rotated
            
            X_augmented.append(X_rotated)
            y_augmented.append(y)
        
        # 3. Small translations (simulate slight position shifts)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in shifts:
            X_shifted = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[3]):  # For each eye channel
                    # Get original image
                    img = X[i, :, :, j]
                    
                    # Define translation matrix
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    
                    # Apply translation
                    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    
                    # Store back
                    X_shifted[i, :, :, j] = shifted
            
            X_augmented.append(X_shifted)
            y_augmented.append(y)
        
        # Combine all augmentations
        X_augmented = np.vstack(X_augmented)
        y_augmented = np.vstack(y_augmented)
        
        # Save augmented data
        if output_filename is None:
            output_filename = os.path.join(
                "eye_tracking_data", 
                f"augmented_data_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        # Save data
        data = {
            'points': y_augmented,
            'eye_features': X_augmented
        }
        
        with open(output_filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Augmentation complete: {original_size} samples → {X_augmented.shape[0]} samples")
        print(f"Augmented data saved to {output_filename}")
        
        # Update current calibration data with augmented data
        self.calibration_eye_features = X_augmented.tolist()
        self.calibration_points = y_augmented.tolist()
        
        return True

    def predict_gaze_position(self, eye_images):
        """Predict gaze position from eye features using trained model"""
        if self.gaze_model is None:
            print("No model loaded for prediction")
            return None
        
        # Reshape input for model
        X = np.expand_dims(eye_images, axis=0)  # Add batch dimension
        
        # Predict normalized coordinates
        prediction = self.gaze_model.predict(X, verbose=0)[0]
        
        # Convert back to screen coordinates
        x = int(prediction[0] * self.screen_width)
        y = int(prediction[1] * self.screen_height)
        
        # Apply smoothing
        if self.last_x is not None and self.last_y is not None:
            x = int(self.smoothing_factor * x + (1 - self.smoothing_factor) * self.last_x)
            y = int(self.smoothing_factor * y + (1 - self.smoothing_factor) * self.last_y)
            # Update last position
            self.last_x, self.last_y = x, y
       
        return x, y

    def handle_dwell_click(self, x, y):
       """Handle dwell click mechanism"""
       current_time = time.time()
       
       # Check if we're starting a new dwell
       if self.dwell_position is None:
           self.dwell_position = (x, y)
           self.dwell_start_time = current_time
           return False
       
       # Check if cursor moved too far from initial position
       distance = np.sqrt((x - self.dwell_position[0])**2 + (y - self.dwell_position[1])**2)
       if distance > self.dwell_threshold:
           # Reset dwell if moved too far
           self.dwell_position = (x, y)
           self.dwell_start_time = current_time
           return False
       
       # Check if dwell time threshold is met
       if current_time - self.dwell_start_time >= self.dwell_time:
           # Perform click and reset dwell
           self.dwell_position = None
           self.dwell_start_time = None
           return True
       
       return False

    def start_control(self):
        """Start cursor control mode"""
        print("Starting cursor control mode...")
        self.control_mode = True
        
        # Create a small window to show status and debug info
        cv2.namedWindow("Eye Tracking Status", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Eye Tracking Status", 400, 300)
        
        # Initialize variables
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        # Start the main loop
        while self.control_mode:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror frame horizontally for more intuitive display
            frame = cv2.flip(frame, 1)
            
            # Extract eye features
            result = self.extract_eye_features(frame)
            
            # Create status display
            status_display = np.zeros((300, 400, 3), dtype=np.uint8)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                current_time = time.time()
                fps = frame_count / (current_time - fps_start_time)
                fps_start_time = current_time
                frame_count = 0
            
            # Display FPS
            cv2.putText(status_display, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If we have eye features, predict gaze
            if result is not None and self.gaze_model is not None:
                _, eye_images, frame_with_eyes = result
                
                # Predict gaze position
                gaze_position = self.predict_gaze_position(eye_images)
                
                if gaze_position is not None:
                    x, y = gaze_position
                    
                    # Move cursor
                    pyautogui.moveTo(x, y)
                    
                    # Handle dwell click
                    if self.handle_dwell_click(x, y):
                        # Perform click
                        pyautogui.click()
                        
                        # Visual feedback for click
                        cv2.putText(status_display, "CLICK!", (150, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    # Display coordinates
                    cv2.putText(status_display, f"Gaze: ({x}, {y})", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display dwell progress if active
                    if self.dwell_start_time is not None:
                        dwell_progress = (time.time() - self.dwell_start_time) / self.dwell_time
                        dwell_progress = min(1.0, dwell_progress)
                        
                        # Draw progress bar
                        bar_width = 300
                        bar_height = 20
                        filled_width = int(bar_width * dwell_progress)
                        
                        cv2.rectangle(status_display, (50, 100), (50 + bar_width, 100 + bar_height), 
                                        (100, 100, 100), -1)
                        cv2.rectangle(status_display, (50, 100), (50 + filled_width, 100 + bar_height), 
                                        (0, 255, 0), -1)
                        cv2.putText(status_display, "Dwell Progress", (120, 95), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display eyes for debugging
                if self.debug:
                    # Left eye
                    left_eye = cv2.resize(eye_images[:, :, 0], (128, 96))
                    left_eye = cv2.cvtColor((left_eye * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    status_display[180:180+96, 20:20+128] = left_eye
                    
                    # Right eye
                    right_eye = cv2.resize(eye_images[:, :, 1], (128, 96))
                    right_eye = cv2.cvtColor((right_eye * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    status_display[180:180+96, 168:168+128] = right_eye
            else:
                cv2.putText(status_display, "No eyes detected", (100, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display status
            cv2.imshow("Eye Tracking Status", status_display)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Toggle debug mode
                self.debug = not self.debug
                print(f"Debug mode: {self.debug}")
            elif key == ord('s'):
                # Adjust smoothing
                self.smoothing_factor = min(0.9, self.smoothing_factor + 0.1)
                print(f"Smoothing factor: {self.smoothing_factor:.1f}")
            elif key == ord('a'):
                # Adjust smoothing
                self.smoothing_factor = max(0.1, self.smoothing_factor - 0.1)
                print(f"Smoothing factor: {self.smoothing_factor:.1f}")
            elif key == ord('+'):
                # Increase dwell time
                self.dwell_time += 0.1
                print(f"Dwell time: {self.dwell_time:.1f} seconds")
            elif key == ord('-'):
                # Decrease dwell time
                self.dwell_time = max(0.2, self.dwell_time - 0.1)
                print(f"Dwell time: {self.dwell_time:.1f} seconds")
        
        # Clean up
        cv2.destroyAllWindows()
        self.control_mode = False

    def run(self):
        """Main function to run the eye tracking interface"""
        print("Eye Tracking Interface")
        print("=====================")
        print("1. Run calibration")
        print("2. Collect additional data")
        print("3. Load existing calibration data")
        print("4. Train model")
        print("5. Perform data augmentation")
        print("6. Start cursor control")
        print("q. Quit")
        
        while True:
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                self.run_calibration()
            elif choice == '2':
                self.collect_additional_data()
            elif choice == '3':
                self.load_calibration_data()
            elif choice == '4':
                self.train_model()
            elif choice == '5':
                self.perform_data_augmentation()
            elif choice == '6':
                if self.gaze_model is None:
                    print("No model loaded. Attempting to load model...")
                    if not self.load_model():
                        print("Failed to load model. Please train or load a model first.")
                        continue
                self.start_control
                self.start_control()
            elif choice.lower() == 'q':
                break
            else:
                print("Invalid choice")
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Eye tracking interface closed")

def main():
   """Main function"""
   try:
       # Create and run eye tracking interface
       eti = EyeTrackingInterface()
       eti.run()
   except Exception as e:
       print(f"Error: {e}")
       import traceback
       traceback.print_exc()

if __name__ == "__main__":
   main()
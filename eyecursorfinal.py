import cv2
import numpy as np
import time
import pyautogui
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import platform

class EyeTrackerCursor:
    def __init__(self):
        # Check if we're on macOS
        self.isMac = platform.system() == 'Darwin'
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6   
        )
        
        # MediaPipe Face Mesh landmarks for eyes
        self.leftEye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.rightEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris landmarks for more precise tracking
        self.leftIris = [474, 475, 476, 477]
        self.rightIris = [469, 470, 471, 472]
        
        # Define the screen size for macOS (handling Retina displays)
        self.screenWidth, self.screenHeight = pyautogui.size()
        
        # For Mac Retina displays, adjust the scaling factor
        if self.isMac:
            try:
                import AppKit
                main_screen = AppKit.NSScreen.mainScreen()
                backingScaleFactor = main_screen.backingScaleFactor()
                if backingScaleFactor > 1.0:
                    print(f"Detected Retina display with scale factor: {backingScaleFactor}")
            except ImportError:
                print("AppKit not available, assuming standard display")
        
        # Initialize calibration data
        self.calibrationData = []
        self.calibrationPoints = []
        
        self.modelX = None
        self.modelY = None
        self.scaler = StandardScaler()
        
        # Enhanced mode settings with Mac-specific parameters
        self.mode = "cursor"  # Default mode: cursor control
        self.blinkThreshold = 0.2  # Adjusted for Mac camera sensitivity
        self.lastBlinkTime = time.time()
        self.blinkCooldown = 0.05  # Increased to prevent accidental clicks
        self.longBlinkThreshold = 1.0  # Seconds for long blink detection
        self.blinkStartTime = None
        
        # Smoothing parameters for cursor movement
        self.smoothingFactor = 0.8  # How much to smooth movement (higher = smoother)
        self.prevX, self.prevY = None, None
        self.cursorSpeed = 0.05  # Reduced for more controlled movement on Mac
        
        # Camera settings
        self.cap = None
        self.cameraIndex = 0  # Default camera
        self.frameWidth = 1280  # Higher resolution for Mac cameras
        self.frameHeight = 720
        
        self.dataDir = os.path.expanduser("~/PATHTOCALIBRATIONDATA")
        os.makedirs(self.dataDir, exist_ok=True)
        
        
    def startCamera(self):
        """Initialize the webcam capture with Mac-optimized settings"""
        self.cap = cv2.VideoCapture(self.cameraIndex)
        
        if not self.cap.isOpened():
            print("Failed to open default camera. Trying alternative...")
            self.cameraIndex = 1
            self.cap = cv2.VideoCapture(self.cameraIndex)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
            

            if self.isMac:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                
            actualWidth = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actualHeight = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera initialized at resolution: {actualWidth}x{actualHeight}")
            
            return True
        else:
            print("Failed to open any camera")
            return False
    
    def calculateEyeAspectRatio(self, eyeLandmarks):
        # Compute the euclidean distances between the vertical eye landmarks
        # Using more points than original for better accuracy
        v1 = np.linalg.norm(eyeLandmarks[1] - eyeLandmarks[5])
        v2 = np.linalg.norm(eyeLandmarks[2] - eyeLandmarks[4])
        v3 = np.linalg.norm(eyeLandmarks[3] - eyeLandmarks[7])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        h = np.linalg.norm(eyeLandmarks[0] - eyeLandmarks[8])
        
        ear = (v1 + v2 + v3) / (3.0 * h)
        
        return ear
    
    def extractEyeFeatures(self, landmarks):
        # Extract full eye contour landmarks
        leftEyeLandmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in self.leftEye])
        rightEyeLandmarks = np.array([[landmarks[i].x, landmarks[i].y] for i in self.rightEye])
        
        # Extract iris landmarks for better tracking
        leftIris = np.array([[landmarks[i].x, landmarks[i].y] for i in self.leftIris])
        rightIris = np.array([[landmarks[i].x, landmarks[i].y] for i in self.rightIris])
        
        # Calculate eye centers using iris center for better precision
        leftEyeCenter = np.mean(leftIris, axis=0)
        rightEyeCenter = np.mean(rightIris, axis=0)
        
        # Calculate eye aspect ratios using more landmarks for accuracy
        leftEar = self.calculateEyeAspectRatio(leftEyeLandmarks[:9])  # Using first 9 points for EAR
        rightEar = self.calculateEyeAspectRatio(rightEyeLandmarks[:9])
        
        # Calculate the distance between iris and eye corners for additional features
        leftIrisToCorner = np.linalg.norm(leftEyeCenter - leftEyeLandmarks[0])
        rightIrisToCorner = np.linalg.norm(rightEyeCenter - rightEyeLandmarks[0])
        
        # New 3D features for head pose estimation
        noseTip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        chin = np.array([landmarks[199].x, landmarks[199].y, landmarks[199].z])
        leftEyeLeft = np.array([landmarks[self.leftEye[0]].x, landmarks[self.leftEye[0]].y, landmarks[self.leftEye[0]].z])
        rightEyeRight = np.array([landmarks[self.rightEye[8]].x, landmarks[self.rightEye[8]].y, landmarks[self.rightEye[8]].z])
        
        # Calculate head pose indicators
        headPitch = np.arctan2(chin[1] - noseTip[1], chin[2] - noseTip[2])
        headYaw = np.arctan2(rightEyeRight[0] - leftEyeLeft[0], rightEyeRight[2] - leftEyeLeft[2])
        
        # Add relative positioning features (ratios rather than absolute positions)
        faceWidth = np.linalg.norm(rightEyeRight - leftEyeLeft)
        leftIrisRelX = (leftEyeCenter[0] - leftEyeLandmarks[0][0]) / faceWidth
        rightIrisRelX = (rightEyeCenter[0] - rightEyeLandmarks[0][0]) / faceWidth
        
        # Return enhanced features
        features = np.concatenate([
            leftEyeCenter,
            rightEyeCenter,
            np.mean(leftIris, axis=0),  # Iris centers
            np.mean(rightIris, axis=0),
            [leftEar, rightEar],
            [leftIrisToCorner, rightIrisToCorner],
            # Head position features to compensate for head movement
            [landmarks[1].x, landmarks[1].y],  # Nose tip
            [landmarks[199].x, landmarks[199].y],  # Chin
            # New head pose features
            [headPitch, headYaw],
            [leftIrisRelX, rightIrisRelX],
            [faceWidth]
        ])
        
        return features
    
    def calibrate(self, numPoints=100):
        if not self.startCamera():
            print("Failed to open camera")
            return False
        
        xPoints = [self.screenWidth * 0.1, self.screenWidth * 0.2, 
                   self.screenWidth * 0.3, self.screenWidth * 0.4, 
                   self.screenWidth * 0.5, self.screenWidth * 0.6,
                   self.screenWidth * 0.7, self.screenWidth * 0.8,
                   self.screenWidth * 0.9, self.screenWidth * 0.95
                   ]
        yPoints = [self.screenHeight * 0.1, self.screenHeight * 0.2, 
                   self.screenHeight * 0.3, self.screenHeight * 0.4, 
                   self.screenHeight * 0.5, self.screenHeight * 0.6, 
                   self.screenHeight * 0.7, self.screenHeight * 0.8, 
                   self.screenHeight * 0.9, self.screenWidth * 0.95
                   ]
        
        self.calibrationPoints = []
        for x in xPoints:
            for y in yPoints:
                if len(self.calibrationPoints) < numPoints:
                    self.calibrationPoints.append((int(x), int(y)))
        
        print(f"Starting calibration with {len(self.calibrationPoints)} points...")
        print("Please follow the cursor and focus on each point when it appears.")
        time.sleep(2)  
        
        for pointIdx, (x, y) in enumerate(self.calibrationPoints):
            calibWindow = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.putText(calibWindow, f"Point {pointIdx+1}/{len(self.calibrationPoints)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(calibWindow, f"Look at position ({x}, {y})", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imshow("Calibration", calibWindow)
            cv2.waitKey(1)
            
            # Move cursor to calibration point
            pyautogui.moveTo(x, y)
            
            countdown = 3
            while countdown > 0:
                calibWindow = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, f"Point {pointIdx+1}/{len(self.calibrationPoints)}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(calibWindow, f"Starting in {countdown}...", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.imshow("Calibration", calibWindow)
                cv2.waitKey(1)
                time.sleep(1)
                countdown -= 1
            
            calibWindow = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.putText(calibWindow, f"Point {pointIdx+1}/{len(self.calibrationPoints)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(calibWindow, "Hold still and focus...", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imshow("Calibration", calibWindow)
            cv2.waitKey(1)
            
            startTime = time.time()
            pointData = []
            
            while time.time() - startTime < 3: 
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.faceMesh.process(frameRgb)
                
                if not results.multi_face_landmarks:
                    continue
                
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extractEyeFeatures(landmarks)
                pointData.append(features)
                
                # Show progress
                elapsed = time.time() - startTime
                progress = min(elapsed / 3.0, 1.0) * 100
                
                calibWindow = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, f"Point {pointIdx+1}/{len(self.calibrationPoints)}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(calibWindow, "Hold still and focus...", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.rectangle(calibWindow, (20, 100), (380, 130), (0, 0, 0), 2)
                cv2.rectangle(calibWindow, (20, 100), (20 + int(360 * progress/100), 130), (0, 255, 0), -1)
                cv2.imshow("Calibration", calibWindow)
                cv2.waitKey(1)
            
            if len(pointData) > 10:  
                pointData = np.array(pointData)
                medianFeatures = np.median(pointData, axis=0)
                distances = np.sum((pointData - medianFeatures)**2, axis=1)
                
                # Keep the best 80% of points
                keepIndices = np.argsort(distances)[:int(len(distances) * 0.8)]
                filteredData = pointData[keepIndices]
                
                # Add the average of filtered data to calibration data
                avgFeatures = np.mean(filteredData, axis=0)
                self.calibrationData.append((avgFeatures, (x, y)))
                
                calibWindow = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, f"Point {pointIdx+1} complete!", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(calibWindow, f"Collected {len(filteredData)} valid samples", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.imshow("Calibration", calibWindow)
                cv2.waitKey(300) 
            else:
                print(f"Failed to collect enough data for point {pointIdx + 1}")
                
                # Update with failure message
                calibWindow = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, f"Point {pointIdx+1} failed!", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(calibWindow, "Not enough valid data collected", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.imshow("Calibration", calibWindow)
                cv2.waitKey(1000)  
        
        cv2.destroyWindow("Calibration")
        
        # Train the models with collected data
        if len(self.calibrationData) >= 6: 
            X = np.array([data[0] for data in self.calibrationData])
            y_x = np.array([data[1][0] for data in self.calibrationData])
            y_y = np.array([data[1][1] for data in self.calibrationData])
            
            XScaled = self.scaler.fit_transform(X)
            
            self.modelX = GradientBoostingRegressor(n_estimators=400, max_depth=18)
            self.modelY = GradientBoostingRegressor(n_estimators=400, max_depth=18)
            
            self.modelX.fit(XScaled, y_x)
            self.modelY.fit(XScaled, y_y)
            
            xPred = self.modelX.predict(XScaled)
            yPred = self.modelY.predict(XScaled)
            
            meanErrorX = np.mean(np.abs(xPred - y_x))
            meanErrorY = np.mean(np.abs(yPred - y_y))
            
            print(f"Calibration complete with {len(self.calibrationData)} points")
            print(f"Average error: X={meanErrorX:.1f}px, Y={meanErrorY:.1f}px")
            
            # Display calibration quality
            quality = "Excellent" if (meanErrorX + meanErrorY) / 2 < 50 else \
                      "Good" if (meanErrorX + meanErrorY) / 2 < 100 else \
                      "Fair" if (meanErrorX + meanErrorY) / 2 < 150 else "Poor"
                      
            print(f"Calibration quality: {quality}")
            return True
        else:
            print("Calibration failed - not enough valid data points")
            return False
    
    

    def saveModels(self, filenamePrefix=None):
        if filenamePrefix is None:
            filenamePrefix = os.path.join(self.dataDir, "eye_tracker_model")
        
        if self.modelX and self.modelY:
            joblib.dump(self.modelX, f"{filenamePrefix}_x.pkl")
            joblib.dump(self.modelY, f"{filenamePrefix}_y.pkl")
            joblib.dump(self.scaler, f"{filenamePrefix}_scaler.pkl")
            print(f"Models saved as {filenamePrefix}_x.pkl and {filenamePrefix}_y.pkl")
            print(f"Models stored in: {self.dataDir}")
            return True
        else:
            print("No models to save. Please calibrate first.")
            return False
    
    def loadModels(self, filenamePrefix=None):
        if filenamePrefix is None:
            filenamePrefix = os.path.join(self.dataDir, "eye_tracker_model")
        
        try:
            self.modelX = joblib.load(f"{filenamePrefix}_x.pkl")
            self.modelY = joblib.load(f"{filenamePrefix}_y.pkl")
            self.scaler = joblib.load(f"{filenamePrefix}_scaler.pkl")
            calibrationFile = os.path.join(self.dataDir, "blink_calibration.pkl")
            calibrationData = joblib.load(calibrationFile)
            
            # Set the calibration parameters
            self.blinkThreshold = calibrationData.get("blinkThreshold", 0.2)
            self.blinkRelativeMode = calibrationData.get("blinkRelativeMode", False)
            self.blinkBaseline = calibrationData.get("blinkBaseline", 0.3)
            self.blinkRelativeThreshold = calibrationData.get("blinkRelativeThreshold", 0.2)
            print("Models loaded successfully")
            return True
        except FileNotFoundError:
            print(f"Model files not found in {self.dataDir}")
            return False
    
    def switchMode(self):
        modes = ["cursor", "scroll", "click", "drag"]
        currentIdx = modes.index(self.mode)
        nextIdx = (currentIdx + 1) % len(modes)
        self.mode = modes[nextIdx]
        
        if self.mode == "cursor":
            self.cursorSpeed = 0.05  
        elif self.mode == "scroll":
            self.cursorSpeed = 0.02  
        elif self.mode == "click":
            self.cursorSpeed = 0.04  
        elif self.mode == "drag":
            self.cursorSpeed = 0.03  
            
        print(f"Mode switched to: {self.mode}")
        return self.mode
    

    def saveBlinkCalibration(self):
        try:
            # Create a dictionary with all calibration parameters
            calibrationData = {
                "blinkThreshold": getattr(self, "blinkThreshold", 0.2),  
                "blinkRelativeMode": getattr(self, "blinkRelativeMode", False),
                "blinkBaseline": getattr(self, "blinkBaseline", 0.3),
                "blinkRelativeThreshold": getattr(self, "blinkRelativeThreshold", 0.2)
            }
            
            calibrationFile = os.path.join(self.dataDir, "blink_calibration.pkl")
            joblib.dump(calibrationData, calibrationFile)
            print(f"Blink calibration saved to {calibrationFile}")
            return True
        except Exception as e:
            print(f"Error saving blink calibration: {e}")
            return False

    def calibrateBlinkThreshold(self):
        if not self.startCamera():
            print("Failed to open camera")
            return False
            
        print("Manual blink calibration. You will indicate when you blink.")
        
        calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calibWindow, "Manual Blink Calibration", (120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calibWindow, "First, we'll record your normal eye state", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calibWindow, "Keep eyes open and look normally", 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calibWindow, "Press SPACE to start recording (5 seconds)", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calibWindow)
        
        while True:
            if cv2.waitKey(1) & 0xFF == 32:  
                break
        
        # Collect baseline EAR values (eyes open)
        baselineEars = []
        
        # Collect normal EAR values for 5 seconds
        startTime = time.time()
        while time.time() - startTime < 5:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(frameRgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extractEyeFeatures(landmarks)
                
                # Get ear values from features based on their position in the array
                leftEar = features[-6] 
                rightEar = features[-5]
                avgEar = (leftEar + rightEar) / 2
                baselineEars.append(avgEar)
                
                progress = min((time.time() - startTime) / 5.0, 1.0) * 100
                calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, "Recording Baseline", (150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calibWindow, "Keep eyes open normally", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Current EAR: {avgEar:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
                cv2.rectangle(calibWindow, (50, 150), (450, 180), (0, 0, 0), 2)
                cv2.rectangle(calibWindow, (50, 150), (50 + int(400 * progress/100), 180), (0, 255, 0), -1)
                cv2.imshow("Calibration", calibWindow)
                cv2.waitKey(1)
        
        calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(calibWindow, "Blink Recording", (150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(calibWindow, "Now we'll record your blinks", 
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calibWindow, "HOLD SPACE when blinking, RELEASE when done", 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(calibWindow, "We need 5 blinks - Press any key to start", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow("Calibration", calibWindow)
        cv2.waitKey(0)
        
        # Collect blink EAR values
        blinkCount = 0
        blinkEarValues = []
        currentBlinkEars = []
        
        while blinkCount < 5:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(frameRgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extractEyeFeatures(landmarks)
                
                # Get ear values
                leftEar = features[-6] 
                rightEar = features[-5]
                avgEar = (leftEar + rightEar) / 2
                
                # Update display
                calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, "Blink Recording", (150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calibWindow, f"Blink count: {blinkCount}/5", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Current EAR: {avgEar:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
                
                # Check if SPACE is pressed (user is blinking)
                key = cv2.waitKey(1) & 0xFF
                if key == 32: 
                    cv2.putText(calibWindow, "BLINKING DETECTED!", 
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    currentBlinkEars.append(avgEar)
                elif len(currentBlinkEars) > 0: 
                    blinkCount += 1
                    minBlinkEar = min(currentBlinkEars)
                    blinkEarValues.append(minBlinkEar)
                    currentBlinkEars = []  
                    
                    calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
                    cv2.putText(calibWindow, "Blink Recorded!", (150, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.putText(calibWindow, f"Blink {blinkCount}/5 captured", 
                            (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    cv2.putText(calibWindow, f"Blink EAR: {minBlinkEar:.4f}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.imshow("Calibration", calibWindow)
                    cv2.waitKey(500) 
                    
                if key == 27:  
                    cv2.destroyWindow("Calibration")
                    return False
                    
                cv2.imshow("Calibration", calibWindow)
        
        # Calculate personalized threshold
        if len(baselineEars) > 10 and len(blinkEarValues) >= 3:
            baselineEars.sort()
            filteredBaseline = baselineEars[int(len(baselineEars)*0.1):int(len(baselineEars)*0.9)]
            avgBaseline = np.mean(filteredBaseline)
            
            # Get average of minimum blink EAR values
            avgBlinkMin = np.mean(blinkEarValues)
            
            # Set threshold halfway between baseline and min blink value
            self.blinkThreshold = (avgBaseline + avgBlinkMin) / 2
            
            # Check if values are very close
            if abs(avgBaseline - avgBlinkMin) < 0.05:
                self.blinkRelativeMode = True
                self.blinkBaseline = avgBaseline
                reductionRatio = avgBlinkMin / avgBaseline
                self.blinkRelativeThreshold = (1 - reductionRatio) * 0.8  # 80% of the observed reduction
                
                print(f"Normal EAR: {avgBaseline:.4f}, Blink EAR: {avgBlinkMin:.4f}")
                print(f"Using relative blink detection with {self.blinkRelativeThreshold:.2f} threshold")
                
                calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, "Calibration Complete", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calibWindow, f"Using relative blink detection", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Normal EAR: {avgBaseline:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Blink EAR: {avgBlinkMin:.4f}", 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Reduction threshold: {self.blinkRelativeThreshold:.2f}", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, "Press any key to continue", 
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                # Standard threshold mode
                print(f"Personalized blink threshold set to: {self.blinkThreshold:.4f}")
                print(f"Normal EAR: {avgBaseline:.4f}, Blink EAR: {avgBlinkMin:.4f}")
                
                calibWindow = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(calibWindow, "Calibration Complete", (120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(calibWindow, f"Blink threshold: {self.blinkThreshold:.4f}", 
                        (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Normal EAR: {avgBaseline:.4f}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, f"Blink EAR: {avgBlinkMin:.4f}", 
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(calibWindow, "Press any key to continue", 
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            self.saveBlinkCalibration()

            cv2.imshow("Calibration", calibWindow)
            cv2.waitKey(0)
            cv2.destroyWindow("Calibration")
            return True
        else:
            print("Blink calibration failed - not enough data")
            return False



    def handleBlink(self, leftEar, rightEar):
        currentTime = time.time()
        avgEar = (leftEar + rightEar) / 2
        print(f"Current EAR: {avgEar:.2f}, Threshold: {self.blinkThreshold}")
        

        if hasattr(self, 'blinkRelativeMode') and self.blinkRelativeMode:
            # Get baseline from recent history (last 30 frames, excluding potential blinks)
            if not hasattr(self, 'earHistory'):
                self.earHistory = []
            self.earHistory.append(avgEar)
            if len(self.earHistory) > 30:
                self.earHistory.pop(0)
            
            # Calculate dynamic baseline (upper percentile to avoid including blinks)
            if len(self.earHistory) >= 10:
                baseline = np.percentile(self.earHistory, 80)
                # Consider it a blink if below percentage of baseline
                isBlink = avgEar < (baseline * (1 - self.blinkRelativeThreshold))
            else:
                isBlink = False
        else:
            isBlink = avgEar < self.blinkThreshold
        

        if isBlink and self.blinkStartTime is None:
            self.blinkStartTime = currentTime
            return None 
        
        elif not isBlink and self.blinkStartTime is not None:
            blinkDuration = currentTime - self.blinkStartTime
            self.blinkStartTime = None
            
            if currentTime - self.lastBlinkTime > self.blinkCooldown:
                self.lastBlinkTime = currentTime
                
                # Different actions based on blink duration
                if blinkDuration < 0.3:  
                    if self.mode == "cursor" or self.mode == "click":
                        # Left click
                        pyautogui.click()
                        print("Click!")
                        return "click"
                    elif self.mode == "drag":
                        pyautogui.mouseDown() if not hasattr(self, 'dragging') or not self.dragging else pyautogui.mouseUp()
                        self.dragging = not getattr(self, 'dragging', False)
                        return "drag_toggle"
                elif 0.3 <= blinkDuration < self.longBlinkThreshold:  
                    if self.mode == "cursor" or self.mode == "click":
                        pyautogui.rightClick()
                        return "right_click"
                else:  
                    # Switch mode on long blink
                    newMode = self.switchMode()
                    return f"mode_switch_{newMode}"
        
        return None
    
    def handleScroll(self, yPosition):
        if self.mode == "scroll":
            centerRegion = 0.3 
            scrollSpeedFactor = 0.5  
            # Calculate normalized position (0 to 1) with center region removed
            normalizedY = yPosition / self.screenHeight
            
            if normalizedY < 0.5 - centerRegion/2:
                # Scroll up with variable speed based on distance from center
                distanceFromCenter = (0.5 - centerRegion/2) - normalizedY
                scrollAmount = int(20 * distanceFromCenter * scrollSpeedFactor)
                pyautogui.scroll(scrollAmount)
                return scrollAmount
            elif normalizedY > 0.5 + centerRegion/2:
                # Scroll down with variable speed based on distance from center
                distanceFromCenter = normalizedY - (0.5 + centerRegion/2)
                scrollAmount = int(-20 * distanceFromCenter * scrollSpeedFactor)
                pyautogui.scroll(scrollAmount)
                return scrollAmount
                
        return 0
    
    def smoothMovement(self, xPred, yPred):
        # Initialize previous position if first movement
        if self.prevX is None:
            self.prevX, self.prevY = xPred, yPred
            return xPred, yPred
        
        # Calculate distance from screen edges
        edgeDistanceX = min(xPred, self.screenWidth - xPred) / (self.screenWidth / 2)
        edgeDistanceY = min(yPred, self.screenHeight - yPred) / (self.screenHeight / 2)
        edgeDistance = min(edgeDistanceX, edgeDistanceY)
        
        # Reduce smoothing near edges for better precision
        adjustedSmoothing = self.smoothingFactor * edgeDistance
        
        # Apply adaptive smoothing (less smoothing near edges)
        smoothX = self.prevX * adjustedSmoothing + xPred * (1 - adjustedSmoothing)
        smoothY = self.prevY * adjustedSmoothing + yPred * (1 - adjustedSmoothing)
        
        # Update previous positions
        self.prevX, self.prevY = smoothX, smoothY
        
        return smoothX, smoothY
    def run(self):
        if not self.cap or not self.cap.isOpened():
            if not self.startCamera():
                print("Failed to open camera")
                return
        
        if not self.modelX or not self.modelY:
            print("Models not trained. Please calibrate first.")
            return
        
        print(f"Eye tracker running on {'macOS' if self.isMac else 'other OS'}.")
        print("Controls: Press 'q' to quit, 'm' to switch modes, 'c' to recalibrate, 'f' for full calibration")
        
        if self.isMac:
            pyautogui.FAILSAFE = False  # Disable corner failsafe for Mac
            
        # Performance tracking
        frameTimes = []
        lastFpsUpdate = time.time()
        
        while True:
            frameStart = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)  
                continue
            
            # Convert to RGB for MediaPipe
            frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(frameRgb)
            
            # Display mode and debug info
            cv2.putText(frame, f"Mode: {self.mode}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extractEyeFeatures(landmarks)
                
                leftEar = features[-6]  
                rightEar = features[-5]
                cv2.putText(frame, f"EAR: {(leftEar + rightEar)/2:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Scale features before prediction
                featuresScaled = self.scaler.transform([features])[0].reshape(1, -1)
                
                # Predict cursor position
                xPred = self.modelX.predict(featuresScaled)[0]
                yPred = self.modelY.predict(featuresScaled)[0]
                
                # Apply smoothing for better experience
                xSmooth, ySmooth = self.smoothMovement(xPred, yPred)
                
                xSmooth = max(0, min(xSmooth, self.screenWidth-1))
                ySmooth = max(0, min(ySmooth, self.screenHeight-1))
                
                # Move cursor with smoother duration
                pyautogui.moveTo(xSmooth, ySmooth, duration=self.cursorSpeed)
                
                # Handle blinks for clicks
                blinkAction = self.handleBlink(leftEar, rightEar)
                if blinkAction:
                    cv2.putText(frame, f"Action: {blinkAction}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Handle scrolling
                scrollAmount = self.handleScroll(ySmooth)
                if scrollAmount != 0:
                    cv2.putText(frame, f"Scroll: {scrollAmount}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw prediction point for debug
                debugX = int(xSmooth * frame.shape[1] / self.screenWidth)
                debugY = int(ySmooth * frame.shape[0] / self.screenHeight)
                cv2.circle(frame, (debugX, debugY), 5, (0, 0, 255), -1)
            
            # Calculate and display FPS
            frameTimes.append(time.time() - frameStart)
            if len(frameTimes) > 30:
                frameTimes.pop(0)
            
            if time.time() - lastFpsUpdate > 1.0:  
                fps = 1.0 / (sum(frameTimes) / len(frameTimes))
                lastFpsUpdate = time.time()          
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Eye Tracker", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.switchMode()
            elif key == ord('s'):
                self.mode = "scroll"
            elif key == ord('c'):
                cv2.destroyAllWindows()
                self.calibrate(9) 
            elif key == ord('f'): 
                print("Starting full calibration from scratch...")
                cv2.destroyAllWindows()
                self.calibrationData = []  
                self.calibrationPoints = []  
                if self.calibrate(16):  
                    self.saveModels()  
                    print("Full calibration complete. Press any key to continue.")
                    cv2.waitKey(0)
                
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        

        if hasattr(self, 'dragging') and self.dragging:
            pyautogui.mouseUp()
            
        print("Eye tracker stopped")

if __name__ == "__main__":
    tracker = EyeTrackerCursor()
    
    # Create a simple startup GUI
    startupWindow = np.ones((300, 500, 3), dtype=np.uint8) * 240
    cv2.putText(startupWindow, "MacOS Eye Tracker", (120, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(startupWindow, "Choose an option:", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(startupWindow, "c - Calibrate (recommended for first use)", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(startupWindow, "l - Load saved calibration", (50, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(startupWindow, "q - Quit", (50, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imshow("Eye Tracker Setup", startupWindow)
    
    choice = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    
    if choice == ord('c'):
        print("Starting calibration...")
        if tracker.calibrate():
            tracker.calibrateBlinkThreshold()
            tracker.saveModels()
            tracker.run()
        else:
            print("Calibration failed")
    elif choice == ord('l'):
        if tracker.loadModels():
            tracker.run()
        else:
            print("Failed to load models. Please calibrate first.")
            retry = input("Would you like to calibrate now? (y/n): ")
            if retry.lower() == 'y':
                if tracker.calibrate():
                    tracker.saveModels()
                    tracker.run()
    
    else:
        print("Exiting program")
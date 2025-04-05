# EyeMouse Controller (macOS)

An AI-powered eye-tracking system for controlling the mouse on macOS using **MediaPipe**, **OpenCV**, and **pyautogui**. Includes intelligent blink detection, mode switching (click/drag/scroll), and calibration for personalized accuracy.

---

## Features

- Real-time eye tracking with **MediaPipe Face Mesh**
- **Blink detection** using EAR (Eye Aspect Ratio) and duration
- Blink-based actions: left click, right click, drag, scroll, and mode switching
- **Smooth cursor movement** with gaze-based velocity adaptation
- **Calibrated EAR thresholds** for robust blink detection per user
- Auto-saves and reuses calibration data across runs
- Scroll mode with vertical gaze control
- Relative EAR-based dynamic blink detection if calibration values are too close
- Smoothing based on gaze distance from center
- Scroll mode splits vertical screen region for scroll speed and direction
- Mode switching by blink duration or hotkey

---

## Blink Detection and Calibration

### Manual Blink Calibration
This system includes a **manual calibration routine** to personalize blink detection using **Eye Aspect Ratio (EAR)**. This is crucial because people's eye shapes and blinking patterns vary.

The calibration is split into two steps:

1. **Baseline EAR Recording** (Eyes Open)
   - You keep your eyes open and look normally.
   - Press `SPACE` to begin a 5-second recording of your natural eye state.
   - The system gathers your typical EAR when not blinking.

2. **Blink EAR Recording** (Active Blink Input)
   - You'll perform **5 blinks manually**, pressing and holding `SPACE` **only while blinking**, then releasing it.
   - The lowest EAR values from these blinks are recorded.

After calibration:

- If your **baseline EAR** and **blink EAR** are very close (e.g., less than 0.05 difference), the system switches to **relative blink mode**, using a dynamic EAR baseline from recent frame history.
- Otherwise, it sets a **static threshold** halfway between your normal and blink EAR values.

All calibration data is saved and reused across sessions. These values are stored in a separate `PATHTOCALIBRATIONDATA/` folder that you must configure.

---

## Blink-Based Interaction

The system interprets **blinks** as different actions based on their duration:

| Blink Type       | Duration               | Action                               |
|------------------|------------------------|--------------------------------------|
| Short Blink      | < 0.3 seconds          | Left Click / Toggle Drag             |
| Medium Blink     | 0.3 - 0.6s (configurable) | Right Click                         |
| Long Blink       | > 0.6s                 | Switch Mode (Cursor / Click / Drag / Scroll) |

### Blink Modes
- `cursor`: Just control the pointer
- `click`: Enables click actions with short blinks
- `drag`: Toggle mouse drag with short blinks
- `scroll`: Use vertical eye position to scroll

You can **switch modes** with a long blink or by pressing `m`.

---

## Scroll with Eye Position

When in `scroll` mode:

- Your **vertical gaze position** controls scroll direction and speed.
- The screen is divided into regions:
  - Middle: No scroll
  - Top/Bottom: Scrolls up/down faster as you look further from center
- Smoothing is applied for a natural feel.

---

## Smooth Cursor Movement

To ensure precision and avoid jitter:

- **Adaptive smoothing** is used, which:
  - Applies more smoothing when your gaze is near screen center.
  - Reduces smoothing near the edges to improve accuracy.
- This helps maintain stable cursor control while allowing fine control near edges (where UI elements often reside).

---

## Requirements

- Python 3.8+
- macOS (tested on macOS Ventura)

### Install Dependencies
Install all dependencies based on the `import` statements in the code:
- mediapipe
- opencv-python
- pyautogui
- numpy
- scikit-learn
- matplotlib

Install them manually using pip:
```bash
pip install mediapipe opencv-python pyautogui numpy scikit-learn matplotlib
```

---

## How to Run

```bash
git clone 
python eyetrackerfinal.py
```

Edit the script to include your path to the calibration data folder (PATHTOCALIBRATIONDATA/).

---

## Hotkeys

- `q`: Quit the program
- `r`: Reset EAR calibration (remove saved EARs)
- `m`: Manually switch modes (cursor → click → drag → scroll)
- `SPACE`: Used during calibration

---

## Project Structure

```
EyeMouse/
├── eyetrackerfinal.py       # Main script with full logic
└── PATHTOCALIBRATIONDATA/   # Folder for storing calibrated EAR values
```

---

## Acknowledgements

- MediaPipe Face Mesh by Google
- Blink detection using EAR inspired by Tereza Soukupová & Jan Čech’s paper
- pyautogui for mouse control
- OpenCV for video processing
- scikit-learn for optional smoothing and filtering logic (if used)
- matplotlib for visual debugging during development (optional)


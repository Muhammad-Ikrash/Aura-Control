# Aura Control

Aura Control is an accessibility-focused gaming controller that turns your webcam into a virtual Xbox 360 gamepad. Built for players with limited hand mobility or amputees, it uses hand and face tracking to map natural gestures to steering, throttle, braking, and button inputs.

## Features

- **Gesture-based driving controls** — steer with one hand, accelerate by pushing your palm toward the camera, brake by pulling away
- **Face-attention safety** — gameplay pauses automatically when you look away or leave the frame
- **Head-nod actions** — nod up or down to press the A button (confirm, nitrous, menu select)
- **Virtual Xbox 360 gamepad** — outputs standard gamepad input recognized by most PC games
- **Live HUD overlay** — real-time preview with a compact car-dashboard instrument cluster
- **Per-user calibration** — tune deadzone, sensitivity, and accel/brake thresholds via the GUI
- **Cross-platform** — Windows GUI app (`AuraController.py`) and Linux CLI (`main_controller.py`)

## ML Models

Aura uses two **MediaPipe Tasks** landmark models for real-time inference. These handle detection only — control mapping is built separately on top.

| Model | File | Role |
|-------|------|------|
| **MediaPipe Hand Landmarker** | `hand_landmarker.task` | Detects up to 2 hands with 21 landmarks each (wrist, knuckles, fingertips) |
| **MediaPipe Face Landmarker** | `face_landmarker.task` | Tracks facial landmarks for gaze direction, head nods, and attention monitoring |

Both models run in `VIDEO` mode via `mediapipe.tasks.python.vision`. Download the `.task` files into the project root:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Custom Control Dataset

Aura does **not** rely on a pre-built gesture-to-gamepad model or off-the-shelf control dataset. Instead, it builds a **custom per-user control profile** from live landmark data:

1. **Custom feature extraction** — each webcam frame is converted into control signals using geometry derived from landmarks:
   - *Steering angle* — angle between wrist (0) and index knuckle (5) on the steering hand
   - *Palm depth* — distance from wrist to the center of all four knuckles (5, 9, 13, 17), used as a gas/brake proxy
   - *Face yaw ratio* — nose-to-cheek distance ratio (landmarks 1, 234, 454) for look-away detection
   - *Face pitch ratio* — nose-chin vs nose-forehead ratio (landmarks 1, 152, 10) for head-nod detection

2. **Temporal smoothing** — rolling buffers (`SmoothValue`) average recent frames to reduce jitter before mapping to gamepad axes.

3. **User calibration profile** — threshold values tuned through the GUI are saved to `aura_config.json`, creating a personalized control dataset for that user's hand size, range of motion, and seating position:

   | Parameter | Purpose |
   |-----------|---------|
   | `STEERING_HAND` | Which hand controls steering (Left / Right) |
   | `STEERING_DEADZONE` | Degrees of hand movement ignored before turning registers |
   | `STEERING_MAX_ANGLE` | Hand tilt for 100% full steering lock |
   | `STEERING_SENSITIVITY` | Steering responsiveness multiplier |
   | `ACCEL_THRESHOLD` | Palm depth at which gas engages |
   | `BRAKE_THRESHOLD` | Palm depth at which brakes engage |

This approach lets each user calibrate the system to their own body instead of fitting everyone into a fixed, pre-trained gesture set.

## How to Use

### Windows (GUI)

1. Install dependencies:
   ```bash
   pip install -r requirements_windows.txt
   ```
2. Download both `.task` model files into the project folder.
3. Run the app:
   ```bash
   python AuraController.py
   ```
4. Adjust the calibration sliders, then click **START CONTROLLER**.
5. Use your steering hand to turn, the other hand for gas/brake, and nod to press A.
6. Press `q` on the camera preview or click **STOP CONTROLLER** to exit.

To build a standalone executable, run `build_windows.bat`. Output is in `dist/AuraController/`.

### Linux (CLI)

1. Install dependencies:
   ```bash
   pip install opencv-python mediapipe evdev
   ```
2. Download both `.task` model files into the project folder.
3. Run with root privileges (required for virtual gamepad creation):
   ```bash
   sudo python main_controller.py
   ```
4. Press `q` on the camera preview to exit safely.

### Control Mapping

| Gesture | Gamepad Output |
|---------|----------------|
| Steering hand tilt | Left stick (X axis) |
| Palm push toward camera | Right trigger (gas) |
| Palm pull away | Left trigger (brake) |
| Head nod up/down | A button |
| Face lost / look away | Start button (pause) |

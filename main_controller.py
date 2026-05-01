import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import sys
from collections import deque

try:
    from evdev import UInput, ecodes as e, AbsInfo, UInputError
except ImportError:
    print("evdev is not installed. Please run: pip install evdev")
    sys.exit(1)

# Configuration
HAND_MODEL = 'hand_landmarker.task'
FACE_MODEL = 'face_landmarker.task'
STEERING_HAND = 0  # 0 for Left, 1 for Right

# Steering Analog Config
# Modify these to tune your steering feel in Blur!
STEERING_DEADZONE = 5.0     # Degrees hand can move before turn is registered natively
STEERING_MAX_ANGLE = 20.0   # Degees turned to hit exactly 100% full lock in game
STEERING_SENSITIVITY = 1.0  # Multiplier for raw angles (2.0 = twice as snappy)

# Speed Analog Config
# Uses rigid palm distance (Wrist to Middle Finger Base) as a proxy for depth
ACCEL_THRESHOLD = 0.15      # Depth value where gas begins to engage (pushing towards camera)
ACCEL_MAX = 0.30            # Depth value for 100% Full Gas Pedal
BRAKE_THRESHOLD = 0.10      # Depth value where brakes begin to engage (pulling away)
BRAKE_MAX = 0.04            # Depth value for 100% Full Brake Pedal

# Face properties
NOD_UP_THRESHOLD = 1.6
NOD_DOWN_THRESHOLD = 0.6
LOOK_AWAY_RATIO_HIGH = 2.0
LOOK_AWAY_RATIO_LOW = 0.5

def distance_2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_angle(p1, p2):
    return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

class SmoothValue:
    def __init__(self, size=5):
        self.q = deque(maxlen=size)
    def update(self, val):
        self.q.append(val)
        return sum(self.q) / len(self.q)

def main():
    print("\n================================================")
    print("Initializing Main Controller (Xbox Gamepad Emulation)...")
    print("================================================\n")
    try:
        cap_events = {
            e.EV_KEY: [e.BTN_SOUTH, e.BTN_EAST, e.BTN_NORTH, e.BTN_WEST, e.BTN_START, e.BTN_SELECT],
            e.EV_ABS: [
                # Analog Stick L-R translates -32768 (full left) to 32767 (full right)
                (e.ABS_X, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
                (e.ABS_Z, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),   # LT (Brake)
                (e.ABS_RZ, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),  # RT (Gas)
            ]
        }
        # Spoof the exact hardware ID for a wired Xbox 360 pad
        gamepad = UInput(cap_events, name='Microsoft X-Box 360 pad', vendor=0x045e, product=0x028e, version=0x0114)
    except UInputError as err:
        print(f"FAILED TO CREATE VIRTUAL CONTROLLER: {err}")
        print("\n[CRITICAL ERROR] You MUST run this script with ROOT privileges (sudo) on Linux!")
        print("Please run:  sudo .venv/bin/python main_controller.py\n")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed to open.")
        return
    
    # Initialize Detectors
    base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL)
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base_options_hand, running_mode=vision.RunningMode.VIDEO, num_hands=2)
    )
    
    base_options_face = python.BaseOptions(model_asset_path=FACE_MODEL)
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base_options_face, running_mode=vision.RunningMode.VIDEO, num_faces=1)
    )

    steer_smoother = SmoothValue(size=3)
    speed_smoother = SmoothValue(size=5)
    
    # Last state trackers to only emit events if changed (evdev requirement)
    last_abs_x, last_abs_z, last_abs_rz = 0, 0, 0
    last_btn_south, last_btn_start = 0, 0

    def emit_gamepad(event_type, event_code, value):
        gamepad.write(event_type, event_code, value)

    def release_all():
        nonlocal last_abs_x, last_abs_z, last_abs_rz, last_btn_south, last_btn_start
        emit_gamepad(e.EV_ABS, e.ABS_X, 0)
        emit_gamepad(e.EV_ABS, e.ABS_Z, 0)
        emit_gamepad(e.EV_ABS, e.ABS_RZ, 0)
        emit_gamepad(e.EV_KEY, e.BTN_SOUTH, 0)
        emit_gamepad(e.EV_KEY, e.BTN_START, 0)
        gamepad.syn()
        last_abs_x = last_abs_z = last_abs_rz = last_btn_south = last_btn_start = 0

    print("Main loop started. Press 'q' on the CV window to exit safely.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        
        hand_res = hand_detector.detect_for_video(mp_img, ts)
        face_res = face_detector.detect_for_video(mp_img, ts)
        
        h, w, _ = frame.shape
        
        # --- FACE LOGIC (Attention & Menu) ---
        paused = False
        btn_south_pressed = 0 # Maps to 'A' button
        
        if not face_res.face_landmarks:
            paused = True
        else:
            face = face_res.face_landmarks[0]
            nose, chin, fh = face[1], face[152], face[10]
            l_cheek, r_cheek = face[234], face[454]
            
            yaw = distance_2d(nose, l_cheek) / max(distance_2d(nose, r_cheek), 0.001)
            pitch = distance_2d(nose, chin) / max(distance_2d(nose, fh), 0.001)
            
            if yaw > LOOK_AWAY_RATIO_HIGH or yaw < LOOK_AWAY_RATIO_LOW:
                paused = True
            
            # Nod to press 'A' (Select/Enter/Nitrous)
            if pitch > NOD_UP_THRESHOLD or pitch < NOD_DOWN_THRESHOLD:
                btn_south_pressed = 1
                
        # Send pause button event
        if paused and last_btn_start == 0:
            release_all()
            emit_gamepad(e.EV_KEY, e.BTN_START, 1)
            gamepad.syn()
            last_btn_start = 1
            
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)
            cv2.putText(frame, "PAUSED: LOOKING AWAY / NO FACE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.imshow('Main Controller', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
        elif not paused and last_btn_start == 1:
            emit_gamepad(e.EV_KEY, e.BTN_START, 0)
            gamepad.syn()
            last_btn_start = 0

        # Send A button event
        if btn_south_pressed != last_btn_south:
            emit_gamepad(e.EV_KEY, e.BTN_SOUTH, btn_south_pressed)
            last_btn_south = btn_south_pressed

        # --- HAND LOGIC (Steer & Speed) ---
        steer_angle = -90.0 
        depth_val = 0.225 # Neutral

        if hand_res.hand_landmarks:
            for idx, hl in enumerate(hand_res.hand_landmarks):
                category = hand_res.handedness[idx][0].category_name
                expected_steering = "Left" if STEERING_HAND == 0 else "Right"
                
                if category == expected_steering: # Steering
                    steer_angle = calculate_angle(hl[0], hl[5])
                    cx1, cy1 = int(hl[0].x * w), int(hl[0].y * h)
                    cx2, cy2 = int(hl[5].x * w), int(hl[5].y * h)
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 4)
                else: # Speed (Depth proxy via Palm length: Wrist to Middle MCP)
                    depth_val = distance_2d(hl[0], hl[9])
                    cx1, cy1 = int(hl[0].x * w), int(hl[0].y * h)
                    cx2, cy2 = int(hl[9].x * w), int(hl[9].y * h)
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 4)
        
        avg_steer = steer_smoother.update(steer_angle)
        avg_depth = speed_smoother.update(depth_val)
        
        # Analog Steering conversion
        delta_angle = (avg_steer + 90.0) * STEERING_SENSITIVITY
        if abs(delta_angle) < STEERING_DEADZONE:
            abs_x = 0
        else:
            direction = 1 if delta_angle > 0 else -1
            magnitude = min(abs(delta_angle) - STEERING_DEADZONE, STEERING_MAX_ANGLE) / STEERING_MAX_ANGLE
            abs_x = int(magnitude * 32767 * direction)
            
        dir_str = "CENTER" if abs_x == 0 else ("RIGHT" if abs_x > 0 else "LEFT")
        
        # Analog Speed/Brake conversion
        abs_rz = 0 # Gas (Right Trigger)
        abs_z = 0  # Brake (Left Trigger)
        
        if avg_depth > ACCEL_THRESHOLD:
            magnitude = min(1.0, (avg_depth - ACCEL_THRESHOLD) / (ACCEL_MAX - ACCEL_THRESHOLD))
            abs_rz = int(magnitude * 255)
            spd_str = f"ACCEL ({int(magnitude*100)}%)"
            cv2.putText(frame, "RT (Gas) Active", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif avg_depth < BRAKE_THRESHOLD:
            # Brake depth gets smaller as hand closes, so reverse calculation
            magnitude = min(1.0, (BRAKE_THRESHOLD - avg_depth) / (BRAKE_THRESHOLD - BRAKE_MAX))
            abs_z = int(magnitude * 255)
            spd_str = f"BRAKE ({int(magnitude*100)}%)"
            cv2.putText(frame, "LT (Brake) Active", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
             spd_str = "COAST"
             
        # Emit Gamepad axes
        if abs_x != last_abs_x:
            emit_gamepad(e.EV_ABS, e.ABS_X, abs_x)
            last_abs_x = abs_x
        if abs_rz != last_abs_rz:
            emit_gamepad(e.EV_ABS, e.ABS_RZ, abs_rz)
            last_abs_rz = abs_rz
        if abs_z != last_abs_z:
            emit_gamepad(e.EV_ABS, e.ABS_Z, abs_z)
            last_abs_z = abs_z
            
        gamepad.syn()

        # Visual Overlay
        pct_steer = int(abs(abs_x) / 32767.0 * 100)
        cv2.putText(frame, f"Steer: {dir_str} {pct_steer}% | Sens: {STEERING_SENSITIVITY}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {spd_str}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if btn_south_pressed:
            cv2.putText(frame, "'A' BUTTON PRESSED", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('Main Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_all()
    hand_detector.close()
    face_detector.close()
    gamepad.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

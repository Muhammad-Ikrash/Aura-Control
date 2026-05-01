import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
from collections import deque
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Configuration
HAND_MODEL = 'hand_landmarker.task'
FACE_MODEL = 'face_landmarker.task'

STEERING_HAND = 0  # 0 for Left, 1 for Right

# Steering properties
LEFT_THRESHOLD = -110
RIGHT_THRESHOLD = -70

# Speed properties
ACCEL_THRESHOLD = 0.15
BRAKE_THRESHOLD = 0.12

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
    print("Initializing Main gesture controller...")
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

    # Step 5: Smoothing Buffers
    steer_smoother = SmoothValue(size=3)
    speed_smoother = SmoothValue(size=5)
    
    # Step 6: Keyboard State Management
    active_keys = {'w': False, 's': False, 'a': False, 'd': False}
    
    def set_key(key_char, state):
        if active_keys[key_char] != state:
            if state:
                keyboard.press(key_char)
            else:
                keyboard.release(key_char)
            active_keys[key_char] = state
            
    def release_all():
        for k in active_keys.keys():
            set_key(k, False)

    pause_latch = False
    nod_latch = False

    print("Main loop started. Press 'q' on the CV window to exit safely and release keys.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        
        # Execute both models
        hand_res = hand_detector.detect_for_video(mp_img, ts)
        face_res = face_detector.detect_for_video(mp_img, ts)
        
        h, w, _ = frame.shape
        
        # --- FACE LOGIC (Attention & Menu) ---
        paused = False
        nodding = False
        
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
            
            if pitch > NOD_UP_THRESHOLD or pitch < NOD_DOWN_THRESHOLD:
                nodding = True
                
        # Handle Paused state immediately
        if paused:
            release_all() # Let go of all simulated driving keys
            if not pause_latch:
                keyboard.press(Key.esc)   # Pause the game (optional depending on your game map)
                keyboard.release(Key.esc)
                pause_latch = True
            
            # Simple visual indicator
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)
            cv2.putText(frame, "PAUSED: LOOKING AWAY / NO FACE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.imshow('Main Controller', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue # Skip hand processing if paused
        else:
            if pause_latch:
                # Optionally hit ESC again to resume if the game unpauses with ESC
                pause_latch = False

        # Menu Nod logic
        if nodding and not nod_latch:
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            nod_latch = True
        elif not nodding:
            nod_latch = False

        # --- HAND LOGIC (Steer & Speed) ---
        steer_angle = -90.0 # Neutral defalt
        depth_val = 0.25 # Neutral default

        if hand_res.hand_landmarks:
            for idx, hl in enumerate(hand_res.hand_landmarks):
                # Get the handedness from MediaPipe ("Left" or "Right")
                category = hand_res.handedness[idx][0].category_name
                expected_steering = "Left" if STEERING_HAND == 0 else "Right"
                
                if category == expected_steering: # Steering hand
                    steer_angle = calculate_angle(hl[0], hl[5])
                    
                    # Draw steering line
                    cx1, cy1 = int(hl[0].x * w), int(hl[0].y * h)
                    cx2, cy2 = int(hl[5].x * w), int(hl[5].y * h)
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 4)

                else: # Speed hand
                    depth_val = distance_2d(hl[4], hl[20])
                    
                    # Draw depth line
                    cx1, cy1 = int(hl[4].x * w), int(hl[4].y * h)
                    cx2, cy2 = int(hl[20].x * w), int(hl[20].y * h)
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 4)
        
        # Smooth Logic
        avg_steer = steer_smoother.update(steer_angle)
        avg_depth = speed_smoother.update(depth_val)
        
        # Steering Rules Engine
        if avg_steer < LEFT_THRESHOLD:
            set_key('a', True); set_key('d', False)
            dir_str = "LEFT ('A')"
        elif avg_steer > RIGHT_THRESHOLD:
            set_key('a', False); set_key('d', True)
            dir_str = "RIGHT ('D')"
        else:
            set_key('a', False); set_key('d', False)
            dir_str = "CENTER"

        # Speed Rules Engine
        if avg_depth > ACCEL_THRESHOLD:
            set_key('w', True); set_key('s', False)
            spd_str = "ACCEL ('W')"
        elif avg_depth < BRAKE_THRESHOLD:
            set_key('w', False); set_key('s', True)
            spd_str = "BRAKE ('S')"
        else:
             set_key('w', False); set_key('s', False)
             spd_str = "COAST"

        # --- VISUAL FEEDBACK (Optional Overlay) ---
        cv2.putText(frame, f"Steer: {dir_str}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"Speed: {spd_str}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        if nodding:
            cv2.putText(frame, "MENU NOD (ENTER)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        cv2.imshow('Main Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up gracefully
    release_all()
    hand_detector.close()
    face_detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

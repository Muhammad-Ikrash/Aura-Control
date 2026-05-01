import sys
import os
import cv2
import threading
import time
import math
import json
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- OS Specific Imports ---
IS_WINDOWS = sys.platform.startswith('win')

CONFIG_FILE = 'aura_config.json'

def get_sys_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

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

class GestureEngine:
    def __init__(self, config_dict):
        self.config = config_dict
        self.running = False
        self.thread = None
        self.gamepad = None

    def _init_gamepad(self):
        if IS_WINDOWS:
            try:
                import vgamepad as vg
                self.gamepad = vg.VX360Gamepad()
                return True
            except ImportError:
                print("Missing vgamepad! Run: pip install vgamepad")
                return False
        else:
            try:
                from evdev import UInput, ecodes as e, AbsInfo, UInputError
                cap_events = {
                    e.EV_KEY: [e.BTN_SOUTH, e.BTN_EAST, e.BTN_NORTH, e.BTN_WEST, e.BTN_START, e.BTN_SELECT, e.BTN_MODE, e.BTN_THUMBL, e.BTN_THUMBR, e.BTN_TL, e.BTN_TR, e.BTN_TL2, e.BTN_TR2],
                    e.EV_ABS: [
                        (e.ABS_X, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
                        (e.ABS_Y, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
                        (e.ABS_Z, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
                        (e.ABS_RX, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
                        (e.ABS_RY, AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
                        (e.ABS_RZ, AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
                        (e.ABS_HAT0X, AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
                        (e.ABS_HAT0Y, AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
                    ]
                }
                self.gamepad = UInput(cap_events, name='Microsoft X-Box 360 pad', vendor=0x045e, product=0x028e, version=0x0114)
                return True
            except Exception as err:
                print(f"Linux UInput Error: {err}\nDid you run as sudo?")
                return False

    def _emit_state(self, abs_x, abs_z, abs_rz, btn_south, btn_start, btn_tl2=0, btn_tr2=0):
        if not self.gamepad: return
        
        if IS_WINDOWS:
            import vgamepad as vg
            self.gamepad.left_joystick(x_value=int(abs_x), y_value=0)
            self.gamepad.left_trigger(value=int(abs_z))
            self.gamepad.right_trigger(value=int(abs_rz))
            
            if btn_south: self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            else: self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            
            if btn_start: self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
            else: self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
            
            self.gamepad.update()
        else:
            from evdev import ecodes as e
            self.gamepad.write(e.EV_ABS, e.ABS_X, abs_x)
            self.gamepad.write(e.EV_ABS, e.ABS_Z, abs_z)
            self.gamepad.write(e.EV_ABS, e.ABS_RZ, abs_rz)
            self.gamepad.write(e.EV_KEY, e.BTN_SOUTH, btn_south)
            self.gamepad.write(e.EV_KEY, e.BTN_START, btn_start)
            self.gamepad.write(e.EV_KEY, e.BTN_TL2, btn_tl2)
            self.gamepad.write(e.EV_KEY, e.BTN_TR2, btn_tr2)
            self.gamepad.syn()

    def start(self, on_fail_callback):
        if not self._init_gamepad():
            on_fail_callback("Failed to initialize Gamepad! (On Linux: Run with sudo. On Win: Install vgamepad)")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.gamepad:
            self._emit_state(0, 0, 0, 0, 0, 0, 0)
            if not IS_WINDOWS:
                self.gamepad.close()
            self.gamepad = None

    def _loop(self):
        cap = cv2.VideoCapture(0)
        hand_model_path = get_sys_path('hand_landmarker.task')
        face_model_path = get_sys_path('face_landmarker.task')

        hand_opts = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=hand_model_path), running_mode=vision.RunningMode.VIDEO, num_hands=2)
        face_opts = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=face_model_path), running_mode=vision.RunningMode.VIDEO, num_faces=1)

        hand_detector = vision.HandLandmarker.create_from_options(hand_opts)
        face_detector = vision.FaceLandmarker.create_from_options(face_opts)

        steer_smoother = SmoothValue(size=3)
        speed_smoother = SmoothValue(size=5)

        last_abs_x, last_abs_z, last_abs_rz = 0, 0, 0
        last_btn_south, last_btn_start = 0, 0
        last_btn_lt, last_btn_rt = 0, 0

        while self.running:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)

            hand_res = hand_detector.detect_for_video(mp_img, ts)
            face_res = face_detector.detect_for_video(mp_img, ts)

            h, w, _ = frame.shape
            
            paused = False
            btn_south_pressed = 0

            # --- FACE LOGIC ---
            if not face_res.face_landmarks:
                paused = True
            else:
                face = face_res.face_landmarks[0]
                pitch = distance_2d(face[1], face[152]) / max(distance_2d(face[1], face[10]), 0.001)
                yaw = distance_2d(face[1], face[234]) / max(distance_2d(face[1], face[454]), 0.001)
                if yaw > 2.0 or yaw < 0.5: paused = True
                if pitch > 1.6 or pitch < 0.6: btn_south_pressed = 1

            if paused:
                self._emit_state(0, 0, 0, 0, 1)
                last_btn_start = 1
                cv2.putText(frame, "PAUSED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow('Aura Controller', frame)
                cv2.waitKey(1)
                continue
            elif last_btn_start == 1:
                last_btn_start = 0

            # --- HAND LOGIC ---
            steer_angle = -90.0
            depth_val = 0.225
            
            conf_steer_hand = "Left" if int(self.config['STEERING_HAND']) == 0 else "Right"

            if hand_res.hand_landmarks:
                for idx, hl in enumerate(hand_res.hand_landmarks):
                    cat = hand_res.handedness[idx][0].category_name
                    if cat == conf_steer_hand:
                        steer_angle = calculate_angle(hl[0], hl[5])
                        cv2.line(frame, (int(hl[0].x*w), int(hl[0].y*h)), (int(hl[5].x*w), int(hl[5].y*h)), (255, 0, 0), 4)
                    else:
                        depth_val = distance_2d(hl[0], hl[9])
                        cv2.line(frame, (int(hl[0].x*w), int(hl[0].y*h)), (int(hl[9].x*w), int(hl[9].y*h)), (255, 0, 255), 4)

            avg_steer = steer_smoother.update(steer_angle)
            avg_depth = speed_smoother.update(depth_val)

            c_dead = float(self.config['STEERING_DEADZONE'])
            c_max = float(self.config['STEERING_MAX_ANGLE'])
            c_sens = float(self.config['STEERING_SENSITIVITY'])
            
            delta_angle = (avg_steer + 90.0) * c_sens
            if abs(delta_angle) < c_dead: abs_x = 0
            else:
                direction = 1 if delta_angle > 0 else -1
                magnitude = min(abs(delta_angle) - c_dead, c_max) / c_max
                abs_x = int(magnitude * 32767 * direction)

            c_acc_thresh = float(self.config['ACCEL_THRESHOLD'])
            c_brk_thresh = float(self.config['BRAKE_THRESHOLD'])

            abs_rz = 0
            abs_z = 0
            btn_rt, btn_lt = 0, 0

            if avg_depth > c_acc_thresh:
                abs_rz = 255
                btn_rt = 1
                cv2.putText(frame, "GAS", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif avg_depth < c_brk_thresh:
                abs_z = 255
                btn_lt = 1
                cv2.putText(frame, "BRAKE", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self._emit_state(abs_x, abs_z, abs_rz, btn_south_pressed, 0, btn_lt, btn_rt)
            
            last_abs_x = abs_x
            last_abs_z = abs_z
            last_abs_rz = abs_rz
            last_btn_south = btn_south_pressed
            last_btn_lt = btn_lt
            last_btn_rt = btn_rt

            cv2.imshow('Aura Controller', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()


class AuraGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aura Game Controller Configuration")
        self.geometry("400x450")
        self.engine = None
        
        self.config = {
            'STEERING_HAND': 0,
            'STEERING_DEADZONE': 5.0,
            'STEERING_MAX_ANGLE': 20.0,
            'STEERING_SENSITIVITY': 1.0,
            'ACCEL_THRESHOLD': 0.15,
            'BRAKE_THRESHOLD': 0.10
        }
        self.load_config()

        self.vars = {}
        row = 0
        
        # UI Generation
        self._build_slider("Steering Deadzone (deg)", 'STEERING_DEADZONE', 0, 20, row)
        row += 1
        self._build_slider("Steering Max Angle (deg)", 'STEERING_MAX_ANGLE', 10, 90, row)
        row += 1
        self._build_slider("Steering Sensitivity", 'STEERING_SENSITIVITY', 0.5, 3.0, row)
        row += 1
        self._build_slider("Accel Threshold", 'ACCEL_THRESHOLD', 0.10, 0.30, row)
        row += 1
        self._build_slider("Brake Threshold", 'BRAKE_THRESHOLD', 0.05, 0.20, row)
        row += 1

        # Hand Radio
        tk.Label(self, text="Steering Hand:").grid(row=row, column=0, sticky='w', pady=10, padx=10)
        self.hand_var = tk.IntVar(value=self.config['STEERING_HAND'])
        tk.Radiobutton(self, text="Left", variable=self.hand_var, value=0, command=self.save_config).grid(row=row, column=1)
        tk.Radiobutton(self, text="Right", variable=self.hand_var, value=1, command=self.save_config).grid(row=row, column=2)
        row += 1

        self.btn_toggle = tk.Button(self, text="START CONTROLLER", bg="green", fg="white", font=("Arial", 12, "bold"), command=self.toggle_engine)
        self.btn_toggle.grid(row=row, column=0, columnspan=3, pady=20, ipadx=50, ipady=10)

    def _build_slider(self, label, key, min_val, max_val, row):
        tk.Label(self, text=label).grid(row=row, column=0, sticky='w', padx=10)
        val_var = tk.DoubleVar(value=self.config[key])
        self.vars[key] = val_var
        
        lbl_val = tk.Label(self, text=f"{val_var.get():.2f}")
        lbl_val.grid(row=row, column=2)

        def on_change(e):
            lbl_val.config(text=f"{val_var.get():.2f}")
            self.save_config()

        sl = ttk.Scale(self, from_=min_val, to=max_val, variable=val_var, command=on_change)
        sl.grid(row=row, column=1)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                self.config.update(json.load(f))

    def save_config(self):
        for k, v in self.vars.items():
            self.config[k] = v.get()
        self.config['STEERING_HAND'] = self.hand_var.get()
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)

    def engine_failed(self, msg):
        messagebox.showerror("Engine Failure", msg)
        self.btn_toggle.config(text="START CONTROLLER", bg="green")

    def toggle_engine(self):
        if self.engine and self.engine.running:
            self.engine.stop()
            self.btn_toggle.config(text="START CONTROLLER", bg="green")
        else:
            self.engine = GestureEngine(self.config)
            self.engine.start(on_fail_callback=self.engine_failed)
            self.btn_toggle.config(text="STOP CONTROLLER", bg="red")

    def destroy(self):
        if self.engine: self.engine.stop()
        super().destroy()

if __name__ == "__main__":
    app = AuraGUI()
    app.mainloop()

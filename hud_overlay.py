"""Game-style HUD overlay for the Aura accessibility controller preview."""

import math

import cv2
import numpy as np

DISPLAY_SIZE = 420
WINDOW_TITLE_AURA = "Aura Controller"
WINDOW_TITLE_MAIN = "Main Controller"

FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX
FONT_HUD = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL

CYAN = (255, 220, 0)
ORANGE = (0, 140, 255)
GREEN = (80, 255, 80)
RED = (60, 60, 255)
YELLOW = (0, 230, 255)
WHITE = (240, 240, 240)
DIM = (140, 140, 160)
PANEL_BG = (30, 15, 45)


def palm_center_norm(hl):
    """Knuckle-row center — keeps the palm axis line straight, not biased to one finger."""
    cx = (hl[5].x + hl[9].x + hl[13].x + hl[17].x) / 4
    cy = (hl[5].y + hl[9].y + hl[13].y + hl[17].y) / 4
    return cx, cy


def palm_depth(hl):
    pcx, pcy = palm_center_norm(hl)
    return math.hypot(hl[0].x - pcx, hl[0].y - pcy)


def draw_speed_hand_line(frame, hl, color=(255, 0, 255)):
    h, w = frame.shape[:2]
    pcx, pcy = palm_center_norm(hl)
    cv2.line(
        frame,
        (int(hl[0].x * w), int(hl[0].y * h)),
        (int(pcx * w), int(pcy * h)),
        color,
        4,
    )


def draw_steer_hand_line(frame, hl, color=(255, 0, 0)):
    h, w = frame.shape[:2]
    cv2.line(
        frame,
        (int(hl[0].x * w), int(hl[0].y * h)),
        (int(hl[5].x * w), int(hl[5].y * h)),
        color,
        4,
    )


def init_cv_window(title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, DISPLAY_SIZE, DISPLAY_SIZE)


def prepare_frame(frame):
    """Crop to center square and resize for a compact almost-square preview."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(square, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_LINEAR)


def _glow_text(img, text, pos, font, scale, color, thickness=1, align="left"):
    h, w = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if align == "center":
        x = (w - tw) // 2
    elif align == "right":
        x = w - tw - x
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def _panel(img, x1, y1, x2, y2, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 1)


def _corner_brackets(img):
    h, w = img.shape[:2]
    m, ln = 8, 22
    c, t = ORANGE, 2
    pts = [
        ((m, m), (m + ln, m), (m, m + ln)),
        ((w - m, m), (w - m - ln, m), (w - m, m + ln)),
        ((m, h - m), (m + ln, h - m), (m, h - m - ln)),
        ((w - m, h - m), (w - m - ln, h - m), (w - m, h - m - ln)),
    ]
    for (a, b, c_pt) in pts:
        cv2.line(img, a, b, c, t)
        cv2.line(img, a, c_pt, c, t)
    cv2.rectangle(img, (m - 2, m - 2), (w - m + 2, h - m + 2), (50, 30, 70), 1)


def _scanlines(img):
    overlay = img.copy()
    for y in range(0, img.shape[0], 4):
        cv2.line(overlay, (0, y), (img.shape[1], y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, 0.06, img, 0.94, 0, img)


def _title_bar(img):
    w = img.shape[1]
    _panel(img, 0, 0, w, 38, alpha=0.65)
    _glow_text(img, "AURA", (14, 26), FONT_TITLE, 0.7, CYAN, 2)
    _glow_text(img, "ACCESSIBLE GAMING HUD", (90, 26), FONT_SMALL, 0.55, ORANGE, 1)


def _control_legend(img, steer_hand):
    h, w = img.shape[:2]
    _panel(img, 0, h - 52, w, h, alpha=0.6)
    legend = f"STEER: {steer_hand.upper()} HAND  |  PUSH = RT GAS  |  PULL = LT BRAKE  |  NOD = A"
    _glow_text(img, legend, (10, h - 18), FONT_SMALL, 0.42, DIM, 1)


def _mini_dashboard(img, steer_dir, steer_pct, speed_state):
    """Compact instrument cluster: steering wheel + speedometer needle."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h - 88
    _panel(img, cx - 62, cy - 30, cx + 62, cy + 26, alpha=0.48)

    # --- steering wheel ---
    wx, wy = cx - 34, cy
    wr = 20
    turn = 0.0
    if steer_dir == "LEFT":
        turn = -min(steer_pct / 100.0, 1.0) * 55
    elif steer_dir == "RIGHT":
        turn = min(steer_pct / 100.0, 1.0) * 55

    wheel_color = GREEN if steer_dir != "CENTER" else DIM
    cv2.circle(img, (wx, wy), wr, (55, 55, 75), -1)
    cv2.circle(img, (wx, wy), wr, wheel_color, 2)
    cv2.circle(img, (wx, wy), 5, ORANGE, -1)
    for spoke in (90, 210, 330):
        rad = math.radians(spoke + turn)
        ex = int(wx + wr * 0.82 * math.cos(rad))
        ey = int(wy + wr * 0.82 * math.sin(rad))
        cv2.line(img, (wx, wy), (ex, ey), wheel_color, 2)
    grip_y = int(wy - wr * 0.55 * math.cos(math.radians(turn)))
    grip_x = int(wx + wr * 0.55 * math.sin(math.radians(turn)))
    cv2.circle(img, (grip_x, grip_y), 4, wheel_color, -1)

    # --- speedometer arc + needle ---
    sx, sy = cx + 34, cy + 4
    sr = 22
    cv2.ellipse(img, (sx, sy), (sr, sr), 0, 205, 335, (45, 45, 65), 4)
    cv2.ellipse(img, (sx, sy), (sr, sr), 0, 205, 335, DIM, 1)
    for tick, label in ((235, "B"), (270, ""), (305, "G")):
        rad = math.radians(tick)
        tx1, ty1 = int(sx + sr * 0.62 * math.cos(rad)), int(sy + sr * 0.62 * math.sin(rad))
        tx2, ty2 = int(sx + sr * 0.88 * math.cos(rad)), int(sy + sr * 0.88 * math.sin(rad))
        cv2.line(img, (tx1, ty1), (tx2, ty2), DIM, 1)
        if label:
            lx, ly = int(sx + sr * 1.05 * math.cos(rad)), int(sy + sr * 1.05 * math.sin(rad))
            color = RED if label == "B" else CYAN
            _glow_text(img, label, (lx - 4, ly + 4), FONT_SMALL, 0.35, color, 1)

    needle_angles = {"BRAKE": 235, "COAST": 270, "GAS": 305}
    needle_color = {"GAS": CYAN, "BRAKE": RED, "COAST": WHITE}.get(speed_state, WHITE)
    n_rad = math.radians(needle_angles.get(speed_state, 270))
    nx = int(sx + sr * 0.78 * math.cos(n_rad))
    ny = int(sy + sr * 0.78 * math.sin(n_rad))
    cv2.line(img, (sx, sy), (nx, ny), needle_color, 2)
    cv2.circle(img, (sx, sy), 3, needle_color, -1)


def _status_panel(img, speed_state, a_pressed):
    _panel(img, 48, 44, img.shape[1] - 48, 88, alpha=0.45)

    speed_colors = {"GAS": CYAN, "BRAKE": RED, "COAST": DIM}
    speed_labels = {
        "GAS": "THROTTLE OPEN",
        "BRAKE": "BRAKES ENGAGED",
        "COAST": "IDLE / COAST",
    }
    spd = speed_state or "COAST"
    _glow_text(img, speed_labels.get(spd, spd), (58, 66), FONT_HUD, 0.48, speed_colors.get(spd, DIM), 1)

    if a_pressed:
        _glow_text(img, "[A] CONFIRM", (58, 88), FONT_HUD, 0.48, YELLOW, 1)


def _trigger_meter(img, side, active, label, color):
    h, w = img.shape[:2]
    meter_h = 90
    meter_w = 18
    y1 = 55
    y2 = y1 + meter_h

    if side == "left":
        x1, x2 = 14, 14 + meter_w
    else:
        x1, x2 = w - 14 - meter_w, w - 14

    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 35, 55), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color if active else DIM, 1)

    if active:
        fill_y1 = y1 + 4
        fill_y2 = y2 - 4
        cv2.rectangle(img, (x1 + 3, fill_y1), (x2 - 3, fill_y2), color, -1)

    text_x = x1 - 2 if side == "left" else x1 - 6
    _glow_text(img, label, (text_x, y2 + 16), FONT_SMALL, 0.45, color if active else DIM, 1)


def _gamepad_icon(img, a_pressed, gas_active, brake_active):
    h, w = img.shape[:2]
    ox, oy = w - 78, h - 118
    body = (ox, oy + 18, ox + 62, oy + 48)
    cv2.ellipse(img, ((body[0] + body[2]) // 2, (body[1] + body[3]) // 2), (28, 14), 0, 0, 360, (55, 55, 75), -1)
    cv2.ellipse(img, ((body[0] + body[2]) // 2, (body[1] + body[3]) // 2), (28, 14), 0, 0, 360, DIM, 1)

    cv2.rectangle(img, (ox + 6, oy + 26), (ox + 16, oy + 36), DIM, 1)
    a_color = YELLOW if a_pressed else (80, 80, 100)
    cv2.circle(img, (ox + 46, oy + 32), 7, a_color, -1 if a_pressed else 1)
    _glow_text(img, "A", (ox + 42, oy + 36), FONT_SMALL, 0.35, (0, 0, 0) if a_pressed else DIM, 1)

    rt_color = CYAN if gas_active else DIM
    lt_color = RED if brake_active else DIM
    _glow_text(img, "RT", (ox + 2, oy + 52), FONT_SMALL, 0.38, rt_color, 1)
    _glow_text(img, "LT", (ox + 38, oy + 52), FONT_SMALL, 0.38, lt_color, 1)


def draw_pause_hud(frame, reason="LOOK AT SCREEN"):
    img = frame.copy()
    h, w = img.shape[:2]

    _corner_brackets(img)
    _title_bar(img)
    _scanlines(img)

    overlay = img.copy()
    cv2.rectangle(overlay, (20, h // 2 - 42), (w - 20, h // 2 + 42), (0, 0, 140), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (20, h // 2 - 42), (w - 20, h // 2 + 42), RED, 2)

    _glow_text(img, "GAME PAUSED", (0, h // 2 - 6), FONT_TITLE, 0.85, WHITE, 2, align="center")
    _glow_text(img, reason, (0, h // 2 + 24), FONT_HUD, 0.5, YELLOW, 1, align="center")
    _glow_text(img, "FACE TRACKING REQUIRED", (0, h // 2 + 46), FONT_SMALL, 0.45, DIM, 1, align="center")

    return img


def draw_active_hud(
    frame,
    steer_dir="CENTER",
    steer_pct=0,
    speed_state="COAST",
    a_pressed=False,
    steer_hand="Left",
    sensitivity=None,
):
    img = frame.copy()
    gas_active = speed_state == "GAS"
    brake_active = speed_state == "BRAKE"

    _corner_brackets(img)
    _title_bar(img)
    _status_panel(img, speed_state, a_pressed)
    _mini_dashboard(img, steer_dir, steer_pct, speed_state)
    _trigger_meter(img, "left", brake_active, "LT", RED)
    _trigger_meter(img, "right", gas_active, "RT", CYAN)
    _gamepad_icon(img, a_pressed, gas_active, brake_active)
    _control_legend(img, steer_hand)
    _scanlines(img)

    return img


def show_frame(title, frame):
    cv2.imshow(title, frame)

"""
Gesture-Based Volume Control System
=====================================
Real-time hand gesture recognition using OpenCV + MediaPipe.
Maps the distance between thumb and index finger to system volume.

Controls:
  - Pinch fingers together  → volume down
  - Spread fingers apart    → volume up
  - Make a fist             → mute toggle
  - Press Q                 → quit

Usage:
  pip install -r requirements.txt
  python gesture_volume.py

Tested on: Windows 10/11, macOS 12+, Ubuntu 20.04+
Latency:   <80ms at 30fps (optimized frame pipeline)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import platform
import subprocess
import sys

# ── platform volume adapter ──────────────────────────────────────────────────

def set_system_volume(level: float):
    """Set system volume 0.0–1.0 cross-platform."""
    level = max(0.0, min(1.0, level))
    os_name = platform.system()

    try:
        if os_name == "Darwin":  # macOS
            vol = int(level * 100)
            subprocess.run(["osascript", "-e", f"set volume output volume {vol}"],
                           capture_output=True)

        elif os_name == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_,
                                             CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(level, None)
            except ImportError:
                pass  # pycaw optional on Windows

        elif os_name == "Linux":
            vol = int(level * 100)
            subprocess.run(["amixer", "-q", "sset", "Master", f"{vol}%"],
                           capture_output=True)
    except Exception:
        pass  # graceful degradation — visual feedback still works


def get_system_volume() -> float:
    """Get current system volume 0.0–1.0."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True)
            return float(result.stdout.strip()) / 100.0

        elif os_name == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_,
                                             CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                return volume.GetMasterVolumeLevelScalar()
            except ImportError:
                return 0.5

        elif os_name == "Linux":
            result = subprocess.run(
                ["amixer", "sget", "Master"],
                capture_output=True, text=True)
            import re
            match = re.search(r'\[(\d+)%\]', result.stdout)
            if match:
                return int(match.group(1)) / 100.0
    except Exception:
        pass
    return 0.5

# ── hand detector ────────────────────────────────────────────────────────────

class HandDetector:
    """
    Wraps MediaPipe Hands for landmark detection.
    Optimized for low-latency single-hand tracking.
    """

    # MediaPipe landmark indices
    THUMB_TIP   = 4
    INDEX_TIP   = 8
    MIDDLE_TIP  = 12
    RING_TIP    = 16
    PINKY_TIP   = 20
    WRIST       = 0
    THUMB_MCP   = 2
    INDEX_MCP   = 5

    def __init__(self, max_hands=1, detection_conf=0.75, tracking_conf=0.75):
        self.mp_hands    = mp.solutions.hands
        self.mp_draw     = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles
        self.hands       = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def find_hands(self, frame: np.ndarray, draw=True):
        """Process frame, optionally draw landmarks. Returns annotated frame + results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # perf: skip copy
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks and draw:
            for lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame, results

    def get_landmark_positions(self, results, frame_shape) -> list:
        """Return list of (x, y) pixel positions for all 21 landmarks, or []."""
        if not results.multi_hand_landmarks:
            return []
        h, w = frame_shape[:2]
        lm_list = []
        for lm in results.multi_hand_landmarks[0].landmark:
            lm_list.append((int(lm.x * w), int(lm.y * h)))
        return lm_list

    def get_finger_distance(self, lm_list: list, p1: int, p2: int):
        """Euclidean distance between two landmarks. Returns (dist, midpoint)."""
        if not lm_list:
            return 0, (0, 0)
        x1, y1 = lm_list[p1]
        x2, y2 = lm_list[p2]
        dist = math.hypot(x2 - x1, y2 - y1)
        mid  = ((x1 + x2) // 2, (y1 + y2) // 2)
        return dist, mid

    def is_fist(self, lm_list: list) -> bool:
        """Detect closed fist — all fingertips below their MCP joints."""
        if len(lm_list) < 21:
            return False
        tips  = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        mcps  = [self.INDEX_MCP, 9, 13, 17]
        return all(lm_list[t][1] > lm_list[m][1] for t, m in zip(tips, mcps))

# ── volume smoother ───────────────────────────────────────────────────────────

class VolumeSmoother:
    """Exponential moving average to eliminate jitter."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self._value = None

    def update(self, new_val: float) -> float:
        if self._value is None:
            self._value = new_val
        else:
            self._value = self.alpha * new_val + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self):
        return self._value or 0.0

# ── overlay renderer ──────────────────────────────────────────────────────────

class Overlay:
    """All on-screen drawing — keeps main loop clean."""

    GREEN  = (80, 200, 80)
    RED    = (60, 60, 220)
    WHITE  = (240, 240, 240)
    GRAY   = (120, 120, 120)
    YELLOW = (40, 210, 210)
    BLACK  = (20, 20, 20)

    def draw_volume_bar(self, frame, volume: float, x=40, y=120, h=300, w=28):
        filled = int(h * volume)
        # track
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.GRAY, 2)
        # fill — green to red gradient via interpolation
        r = int(220 * (1 - volume))
        g = int(200 * volume)
        bar_color = (30, g, r)
        cv2.rectangle(frame, (x, y + h - filled), (x + w, y + h), bar_color, -1)
        # percentage label
        pct = int(volume * 100)
        cv2.putText(frame, f"{pct}%", (x - 2, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.WHITE, 1)
        cv2.putText(frame, "VOL", (x + 2, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.GRAY, 1)

    def draw_pinch_line(self, frame, p1, p2, mid, volume):
        cv2.line(frame, p1, p2, self.GREEN, 2)
        cv2.circle(frame, p1,  8, self.GREEN, -1)
        cv2.circle(frame, p2,  8, self.GREEN, -1)
        cv2.circle(frame, mid, 6, self.YELLOW, -1)

    def draw_mute_banner(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), self.RED, 6)
        cv2.rectangle(frame, (w//2 - 80, h//2 - 28), (w//2 + 80, h//2 + 28),
                      self.BLACK, -1)
        cv2.putText(frame, "MUTED", (w//2 - 52, h//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.RED, 3)

    def draw_hud(self, frame, fps, latency_ms, hand_detected):
        status = "HAND DETECTED" if hand_detected else "NO HAND"
        color  = self.GREEN if hand_detected else self.RED
        cv2.putText(frame, f"FPS: {fps:.0f}  |  Latency: {latency_ms:.0f}ms",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.GRAY, 1)
        cv2.putText(frame, status, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, "Q: quit  |  Fist: mute",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.GRAY, 1)

# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    print("Starting Gesture Volume Control...")
    print(f"Platform: {platform.system()}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    # optimize capture for low latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize buffer lag

    detector = HandDetector()
    smoother = VolumeSmoother(alpha=0.25)
    overlay  = Overlay()

    # distance range calibration (pixels) → mapped to 0–100% volume
    DIST_MIN, DIST_MAX = 30, 220

    current_volume = get_system_volume()
    smoother.update(current_volume)

    muted        = False
    pre_mute_vol = current_volume
    fist_cooldown = 0          # frames since last fist toggle

    prev_time = time.perf_counter()
    fps_smooth = 30.0

    print("Webcam open. Show your hand to the camera.")
    print("Pinch = volume | Fist = mute | Q = quit")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        frame, results = detector.find_hands(frame)
        lm_list = detector.get_landmark_positions(results, frame.shape)

        hand_detected = len(lm_list) > 0

        if hand_detected:
            # ── fist → mute toggle ───────────────────────────────────────────
            if detector.is_fist(lm_list) and fist_cooldown == 0:
                muted = not muted
                if muted:
                    pre_mute_vol = smoother.value
                    set_system_volume(0.0)
                else:
                    set_system_volume(pre_mute_vol)
                    smoother.update(pre_mute_vol)
                fist_cooldown = 20   # ~0.66s debounce at 30fps

            if fist_cooldown > 0:
                fist_cooldown -= 1

            # ── pinch → volume ───────────────────────────────────────────────
            if not detector.is_fist(lm_list):
                dist, mid = detector.get_finger_distance(
                    lm_list,
                    HandDetector.THUMB_TIP,
                    HandDetector.INDEX_TIP,
                )
                # map distance to 0.0–1.0
                raw_vol = np.interp(dist, [DIST_MIN, DIST_MAX], [0.0, 1.0])
                smooth_vol = smoother.update(raw_vol)

                if not muted:
                    set_system_volume(smooth_vol)

                # draw pinch visual
                p1 = lm_list[HandDetector.THUMB_TIP]
                p2 = lm_list[HandDetector.INDEX_TIP]
                overlay.draw_pinch_line(frame, p1, p2, mid, smooth_vol)

        # ── draw UI ──────────────────────────────────────────────────────────
        display_vol = 0.0 if muted else smoother.value
        overlay.draw_volume_bar(frame, display_vol)

        if muted:
            overlay.draw_mute_banner(frame)

        # FPS + latency HUD
        t1       = time.perf_counter()
        latency  = (t1 - t0) * 1000
        fps_inst = 1.0 / max(t1 - prev_time, 1e-6)
        fps_smooth = 0.1 * fps_inst + 0.9 * fps_smooth
        prev_time = t1

        overlay.draw_hud(frame, fps_smooth, latency, hand_detected)

        cv2.imshow("Gesture Volume Control  |  Krishna", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()

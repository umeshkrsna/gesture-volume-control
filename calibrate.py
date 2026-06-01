"""
calibrate.py — Hand distance calibration tool
==============================================
Run this before gesture_volume.py if your hand size or
camera distance is different from the defaults.

It prints the DIST_MIN and DIST_MAX values to paste into
gesture_volume.py lines 160-161.

Usage:
  python calibrate.py
"""

import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def get_dist(lm, p1, p2, w, h):
    x1 = int(lm[p1].x * w); y1 = int(lm[p1].y * h)
    x2 = int(lm[p2].x * w); y2 = int(lm[p2].y * h)
    return math.hypot(x2-x1, y2-y1)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

    min_dist, max_dist = float('inf'), 0.0
    recorded = []

    print("CALIBRATION MODE")
    print("1. Pinch fingers fully CLOSED  (minimum)")
    print("2. Spread fingers fully OPEN   (maximum)")
    print("3. Press S to save a reading | Q to finish\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        dist = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                                   mp_hands.HAND_CONNECTIONS)
            dist = get_dist(lm, 4, 8, w, h)
            cv2.putText(frame, f"Distance: {dist:.1f}px",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,220,80), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60,60,220), 2)

        if recorded:
            cv2.putText(frame, f"Readings: {len(recorded)}  |  Min={min(recorded):.0f}  Max={max(recorded):.0f}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.putText(frame, "S=save reading  |  Q=done",
                    (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and dist is not None:
            recorded.append(dist)
            print(f"  Recorded: {dist:.1f}px  (total {len(recorded)} readings)")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if recorded:
        lo = max(0, min(recorded) - 10)
        hi = max(recorded) + 10
        print(f"\n── Calibration result ──")
        print(f"DIST_MIN = {lo:.0f}")
        print(f"DIST_MAX = {hi:.0f}")
        print(f"\nPaste these values into gesture_volume.py line 160-161.")
    else:
        print("No readings recorded.")

if __name__ == "__main__":
    main()

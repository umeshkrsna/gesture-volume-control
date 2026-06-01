# Gesture-Based Volume Control System

Real-time hand gesture recognition that maps finger movements to system volume — no hardware, no touch required. Built with Python, OpenCV, and MediaPipe.

## Demo

| Gesture | Action |
|---------|--------|
| 👌 Pinch close | Volume down |
| 🤏 Spread apart | Volume up |
| ✊ Fist | Mute / Unmute toggle |
| Q key | Quit |

## Performance

- **Input latency:** <80ms at 30fps
- **Optimization:** Single-buffer capture, EMA smoothing, read-only RGB conversion
- **Detection confidence:** 75% threshold (tunable)

## How it works

```
Webcam frame
    ↓
MediaPipe Hands  →  21 hand landmarks (x, y, z)
    ↓
Thumb tip (4) + Index tip (8) distance
    ↓
np.interp(dist, [30, 220], [0.0, 1.0])   ← maps pixels to volume
    ↓
Exponential Moving Average smoother (α=0.25)
    ↓
Platform volume API  (macOS osascript / Windows pycaw / Linux amixer)
```

## Setup

```bash
git clone https://github.com/umeshkrsna/gesture-volume-control.git
cd gesture-volume-control
pip install -r requirements.txt
python gesture_volume.py
```

**Windows only** — install pycaw for native volume control:
```bash
pip install pycaw comtypes
```

## Calibration

If detection feels off (hand too close/far from camera):

```bash
python calibrate.py
```

Follow the on-screen instructions — pinch fully closed, spread fully open, press S to record readings. It outputs `DIST_MIN` and `DIST_MAX` values to paste into `gesture_volume.py`.

## File structure

```
gesture-volume-control/
├── gesture_volume.py   # main app — hand detection + volume mapping
├── calibrate.py        # calibration utility
├── requirements.txt
└── README.md
```

## Tech stack

- **Python 3.9+**
- **OpenCV** — webcam capture, frame rendering, UI overlay
- **MediaPipe Hands** — 21-point hand landmark detection
- **NumPy** — distance interpolation
- **osascript / pycaw / amixer** — platform volume APIs

## Key classes

| Class | Purpose |
|-------|---------|
| `HandDetector` | Wraps MediaPipe, exposes landmark positions, fist detection |
| `VolumeSmoother` | Exponential moving average — eliminates jitter |
| `Overlay` | All CV2 drawing — volume bar, pinch line, HUD, mute banner |

## Resume context

Built as part of a computer vision exploration project to demonstrate real-time gesture recognition without specialized hardware. Optimized frame pipeline keeps latency under 80ms through single-buffer capture and efficient RGB conversion.

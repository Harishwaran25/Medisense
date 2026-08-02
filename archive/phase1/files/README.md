# 🏥 MediSense — Phase 1: Fall Detection

> **OpenCV + MediaPipe Pose | Patient Monitoring Robot**

---

## What This Does

This script watches a person through a camera and detects if they have **fallen down** and become **motionless** — triggering an alert after 3 seconds of stillness.

```
Camera Feed → MediaPipe Pose → Fall Logic → Status Display → Alert
```

---

## Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

---

## Run It

```bash
# Using your webcam
python fall_detection.py

# Using a video file
python fall_detection.py --video myvideo.mp4
```

---

## How the Fall Detection Works

MediaPipe detects **33 body landmarks** in real time.

We track two key body parts:
- **Shoulders** (landmarks 11 & 12)
- **Hips** (landmarks 23 & 24)

| Situation | Shoulder Y vs Hip Y |
|-----------|-------------------|
| Standing  | Big difference (shoulders much higher) |
| Fallen    | Small difference (both at same level) |

If the ratio drops below the threshold AND the person stays still for **3 seconds** → ALERT triggered.

---

## States

| State | Meaning | Color |
|-------|---------|-------|
| NORMAL | Person is upright | 🟢 Green |
| WARNING | Possible fall detected | 🟠 Orange |
| ALERT | Fall + motionless 3s | 🔴 Red |

---

## Project Roadmap

- [x] **Phase 1** — Fall Detection (OpenCV + MediaPipe) ← YOU ARE HERE
- [ ] **Phase 2** — Pain/Emotion Detection (Hugging Face)
- [ ] **Phase 3** — Webots Simulation + Robot Navigation
- [ ] **Phase 4** — Decision Logic + Voice Alert + Dashboard
- [ ] **Phase 5** — Polish + GitHub Portfolio

---

## Key Concepts You're Learning

- `cv2.VideoCapture()` — reading camera/video frames
- `mediapipe.solutions.pose` — real-time human pose estimation
- Landmark coordinate math — turning body positions into decisions
- State machines — NORMAL → WARNING → ALERT logic
- OpenCV drawing — bounding boxes, text overlays, transparency

---

*MediSense | Built as a unique robotics + CV + ML portfolio project*

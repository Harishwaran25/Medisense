"""
MediSense - Phase 1: Fall Detection
====================================
Uses OpenCV + MediaPipe Pose to detect if a person has fallen.
Run this script with a webcam or a video file.

Install dependencies:
    pip install opencv-python mediapipe numpy

Usage:
    python fall_detection.py                  # uses webcam
    python fall_detection.py --video test.mp4 # uses video file
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FALL_THRESHOLD_RATIO = 0.6   # hip-to-shoulder height ratio below this = possible fall
STILLNESS_SECONDS    = 3.0   # seconds of no movement after fall = alert
MOVEMENT_THRESHOLD   = 0.02  # minimum landmark movement to count as "moving"

# ──────────────────────────────────────────────
# COLORS  (BGR)
# ──────────────────────────────────────────────
COLOR_NORMAL  = (0, 220, 100)   # green
COLOR_WARNING = (0, 165, 255)   # orange
COLOR_ALERT   = (0, 0, 220)     # red
COLOR_TEXT    = (255, 255, 255)
COLOR_BG      = (20, 20, 30)


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def get_landmark_point(landmarks, idx, w, h):
    """Return (x, y) pixel position of a landmark."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def is_fallen(landmarks, frame_height):
    """
    Fall detection logic:
    - Get average Y of shoulders and average Y of hips
    - If hips are near the same height as shoulders (both near bottom),
      the person is likely lying down (fallen).
    - Returns (fallen: bool, ratio: float)
    """
    # Landmark indices (MediaPipe Pose)
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP       = 23
    RIGHT_HIP      = 24

    try:
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        hip_y      = (landmarks[LEFT_HIP].y      + landmarks[RIGHT_HIP].y)      / 2

        # When standing: shoulder_y << hip_y  (shoulders higher up = smaller y)
        # When fallen:   shoulder_y ≈ hip_y   (both at similar height)
        ratio = abs(shoulder_y - hip_y)

        fallen = ratio < FALL_THRESHOLD_RATIO * 0.3
        return fallen, ratio
    except Exception:
        return False, 1.0


def landmark_movement(prev_lms, curr_lms):
    """Average movement of all landmarks between two frames."""
    if prev_lms is None or curr_lms is None:
        return 1.0
    diffs = []
    for p, c in zip(prev_lms, curr_lms):
        diffs.append(abs(p.x - c.x) + abs(p.y - c.y))
    return np.mean(diffs)


def draw_status_box(frame, status, color, ratio, still_time):
    """Draw a semi-transparent status panel on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background panel
    cv2.rectangle(overlay, (10, 10), (360, 130), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Status dot
    cv2.circle(frame, (35, 38), 12, color, -1)

    # Status text
    cv2.putText(frame, status, (55, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

    # Ratio bar
    bar_val = max(0, min(1, ratio / 0.3))
    cv2.rectangle(frame, (15, 60), (355, 80), (60, 60, 60), -1)
    cv2.rectangle(frame, (15, 60), (15 + int(340 * bar_val), 80), color, -1)
    cv2.putText(frame, f"Body alignment ratio: {ratio:.3f}", (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    if still_time > 0:
        cv2.putText(frame, f"Still for: {still_time:.1f}s", (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1)


def draw_alert_banner(frame):
    """Flash a red FALL DETECTED banner."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2 - 50), (w, h//2 + 50), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "⚠  FALL DETECTED — SENDING ALERT", (w//2 - 280, h//2 + 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main(video_source):
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        return

    print("[MediSense] Starting Fall Detection... Press Q to quit.")

    prev_landmarks  = None
    fall_start_time = None
    alert_triggered = False
    state           = "NORMAL"

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            still_time = 0.0
            ratio      = 0.3  # default (standing)

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                # ── Fall detection ──
                fallen, ratio = is_fallen(lms, h)

                # ── Stillness check ──
                movement = landmark_movement(prev_landmarks, lms)
                is_still = movement < MOVEMENT_THRESHOLD

                if fallen:
                    if fall_start_time is None:
                        fall_start_time = time.time()
                    still_time = time.time() - fall_start_time

                    if still_time >= STILLNESS_SECONDS:
                        state = "ALERT"
                        if not alert_triggered:
                            alert_triggered = True
                            print(f"[ALERT] 🚨 Patient has fallen and is motionless for {still_time:.1f}s!")
                    else:
                        state = "WARNING"
                else:
                    fall_start_time = None
                    alert_triggered = False
                    state = "NORMAL"

                prev_landmarks = lms

                # ── Draw skeleton ──
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 200), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(100, 200, 255), thickness=2)
                )

            # ── Status UI ──
            color_map = {
                "NORMAL":  COLOR_NORMAL,
                "WARNING": COLOR_WARNING,
                "ALERT":   COLOR_ALERT,
            }
            status_map = {
                "NORMAL":  "STATUS: NORMAL",
                "WARNING": "STATUS: POSSIBLE FALL",
                "ALERT":   "STATUS: FALL ALERT!",
            }

            color = color_map[state]
            draw_status_box(frame, status_map[state], color, ratio, still_time)

            if state == "ALERT":
                draw_alert_banner(frame)

            # ── FPS counter ──
            cv2.putText(frame, "MediSense v1.0 | Phase 1", (w - 280, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("MediSense - Fall Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[MediSense] Session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediSense Fall Detection")
    parser.add_argument("--video", type=str, default=0,
                        help="Path to video file (default: webcam)")
    args = parser.parse_args()

    source = args.video if args.video != 0 else 0
    main(source)

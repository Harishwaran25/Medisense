"""
MediSense - Phase 2 (Improved): Pain & Emotion Detection
==========================================================
Uses a better Hugging Face model + combined pain scoring
to accurately detect patient distress from facial expressions.

Install:
    pip install transformers torch Pillow opencv-python

Run:
    python3 emotion_detection.py
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import time
import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

# Better model — trained on FER2013 dataset, more accurate
MODEL_NAME = "dima806/facial_emotions_image_detection"

# Pain score = weighted sum of these emotions
PAIN_WEIGHTS = {
    "sad":      0.7,
    "fear":     0.8,
    "disgust":  0.9,
    "angry":    0.6,
    "surprise": 0.3,
    "happy":    0.0,
    "neutral":  0.0,
}

PAIN_SCORE_THRESHOLD = 0.45   # above this = WARNING
CRITICAL_THRESHOLD   = 0.65   # above this = CRITICAL
PAIN_HOLD_SECONDS    = 4.0    # must persist before nurse alert

# ──────────────────────────────────────────────
# COLORS (BGR)
# ──────────────────────────────────────────────
COLOR_NORMAL   = (0, 200, 80)
COLOR_WARNING  = (0, 165, 255)
COLOR_CRITICAL = (0, 0, 210)
COLOR_TEXT     = (255, 255, 255)

alert_log = []

def log_event(event, emotion, pain_score):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {"time": ts, "event": event, "emotion": emotion, "score": pain_score}
    alert_log.append(entry)
    print(f"[ALERT] {ts} | {event} | {emotion} | Pain Score: {pain_score:.2f}")

def calculate_pain_score(predictions):
    """Combine all emotion scores into a single pain score."""
    pain_score = 0.0
    for pred in predictions:
        label  = pred["label"].lower()
        conf   = pred["score"]
        weight = PAIN_WEIGHTS.get(label, 0.0)
        pain_score += weight * conf
    return min(pain_score, 1.0)

def draw_panel(frame, top_emotion, pain_score, state, pain_duration, all_preds):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 210), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    color = (COLOR_NORMAL if state == "NORMAL" else
             COLOR_WARNING if state == "WARNING" else COLOR_CRITICAL)

    cv2.circle(frame, (35, 38), 12, color, -1)
    cv2.putText(frame, f"STATUS: {state}", (55, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Expression: {top_emotion.upper()}", (15, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

    bar_w = int(390 * pain_score)
    cv2.rectangle(frame, (15, 90), (405, 108), (50, 50, 50), -1)
    cv2.rectangle(frame, (15, 90), (15 + bar_w, 108), color, -1)
    cv2.putText(frame, f"Pain Score: {pain_score:.0%}", (15, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

    y_start = 148
    for pred in all_preds[:4]:
        lbl = pred["label"].lower()
        scr = pred["score"]
        bw  = int(150 * scr)
        col = COLOR_CRITICAL if PAIN_WEIGHTS.get(lbl, 0) > 0.5 else (120, 120, 120)
        cv2.rectangle(frame, (15, y_start), (15 + bw, y_start + 10), col, -1)
        cv2.putText(frame, f"{lbl[:7]}  {scr:.0%}", (175, y_start + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT, 1)
        y_start += 14

    if pain_duration > 0:
        cv2.putText(frame, f"Duration: {pain_duration:.1f}s",
                    (15, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WARNING, 1)

def draw_face_box(frame, x, y, w, h, color):
    t, l = 2, 22
    pts  = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (px,py),(dx,dy) in zip(pts,dirs):
        cv2.line(frame,(px,py),(px+dx*l,py),color,t)
        cv2.line(frame,(px,py),(px,py+dy*l),color,t)

def draw_nurse_alert(frame, emotion, score):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-70), (w, h), (0, 0, 150), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame,
        f"  NURSE ALERT  |  Patient Distress: {emotion.upper()}  |  Score: {score:.0%}",
        (15, h-22), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2)

def draw_log(frame, logs):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-290,10), (w-10,140), (15,15,25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "ALERT LOG", (w-280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130,130,130), 1)
    for i, e in enumerate(reversed(logs[-4:])):
        cv2.putText(frame, f"{e['time']}  {e['emotion']}  {e['score']:.0%}",
                    (w-280, 50+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WARNING, 1)

def main():
    print("[MediSense] Loading improved emotion model...")
    print("[MediSense] First run downloads ~400MB — please wait...")

    pipe = pipeline(
        "image-classification",
        model=MODEL_NAME,
        top_k=7
    )
    print("[MediSense] Model ready!")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    pain_start    = None
    nurse_alerted = False
    state         = "NORMAL"
    top_emotion   = "neutral"
    pain_score    = 0.0
    all_preds     = []
    frame_count   = 0

    print("[MediSense] Running! Press Q to quit.")
    print("[MediSense] TIP: Try frowning or looking sad/scared to trigger WARNING")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_count += 1

        if frame_count % 8 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80)
            )

            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])
                face_img = frame[y:y+fh, x:x+fw]
                pil_img  = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                try:
                    all_preds   = pipe(pil_img)
                    top_emotion = all_preds[0]["label"].lower()
                    pain_score  = calculate_pain_score(all_preds)
                except Exception as e:
                    print(f"[WARN] {e}")

                # State machine
                if pain_score >= CRITICAL_THRESHOLD:
                    if pain_start is None:
                        pain_start = time.time()
                    duration = time.time() - pain_start
                    if duration >= PAIN_HOLD_SECONDS:
                        state = "CRITICAL"
                        if not nurse_alerted:
                            nurse_alerted = True
                            log_event("NURSE ALERT", top_emotion, pain_score)
                    else:
                        state = "WARNING"
                elif pain_score >= PAIN_SCORE_THRESHOLD:
                    pain_start    = None
                    nurse_alerted = False
                    state         = "WARNING"
                else:
                    pain_start    = None
                    nurse_alerted = False
                    state         = "NORMAL"

                box_color = (COLOR_NORMAL if state == "NORMAL" else
                             COLOR_WARNING if state == "WARNING" else COLOR_CRITICAL)
                draw_face_box(frame, x, y, fw, fh, box_color)
                cv2.putText(frame, top_emotion,
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        pain_dur = (time.time() - pain_start) if pain_start else 0.0
        draw_panel(frame, top_emotion, pain_score, state, pain_dur, all_preds)

        if alert_log:
            draw_log(frame, alert_log)
        if state == "CRITICAL":
            draw_nurse_alert(frame, top_emotion, pain_score)

        cv2.putText(frame, "MediSense v1.0 | Phase 2",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,80,80), 1)

        cv2.imshow("MediSense - Pain Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[MediSense] Session ended. Total alerts: {len(alert_log)}")
    for e in alert_log:
        print(f"  {e['time']} | {e['emotion']} | Score: {e['score']:.0%}")

if __name__ == "__main__":
    main()

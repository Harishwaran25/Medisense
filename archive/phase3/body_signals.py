"""
MediSense - Phase 3 (Professional UI)
=======================================
Clean, clinical, professional overlay design.
All movement bugs fixed. No fake vitals.

Detects:
  1. Agitation / Restlessness
  2. Posture Change (sit up / roll)
  3. Unconsciousness (sustained stillness)
  4. Bed Boundary Violation

Install:
    pip install opencv-python mediapipe numpy

Run:
    python3 body_signals.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
from collections import deque

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
AGITATION_WINDOW_SEC  = 8.0
AGITATION_HIGH        = 0.028
AGITATION_MILD        = 0.008
UNCONSCIOUS_SEC       = 15.0
MOVEMENT_THRESHOLD    = 0.005
SITTING_UP_THRESHOLD  = 0.12
ROLLING_THRESHOLD     = 0.10
BED_MARGIN            = 0.12
ALERT_COOLDOWN_SEC    = 20

# Body landmarks — NO nose (index 0) to fix "???" bug
BODY_POINTS = [11, 12, 13, 14, 15, 16, 23, 24]

# ──────────────────────────────────────────────
# DESIGN SYSTEM — Clinical Dark Theme (BGR)
# ──────────────────────────────────────────────
C_BG       = (18,  20,  26)
C_PANEL    = (28,  32,  42)
C_BORDER   = (45,  55,  70)
C_OK       = (80,  200, 120)
C_WARN     = (40,  160, 240)
C_CRIT     = (60,  60,  220)
C_TEXT_PRI = (220, 225, 235)
C_TEXT_SEC = (110, 120, 140)
C_ACCENT   = (140, 210, 255)
C_SKELETON = (60,  200, 160)
C_SKEL_CON = (40,  100, 180)
C_BED_LINE = (60,  80,  120)

# ──────────────────────────────────────────────
# ALERT LOG
# ──────────────────────────────────────────────
alert_log   = []
last_alerts = {}

def log_event(event, detail, severity):
    now = time.time()
    if now - last_alerts.get(event, 0) < ALERT_COOLDOWN_SEC:
        return
    last_alerts[event] = now
    ts    = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {"time": ts, "event": event, "detail": detail, "severity": severity}
    alert_log.append(entry)
    print(f"[{severity}] {ts} | {event} | {detail}")


# ──────────────────────────────────────────────
# DRAWING PRIMITIVES
# ──────────────────────────────────────────────
def fill_alpha(frame, x1, y1, x2, y2, color, alpha=0.85):
    ov = frame.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

def border(frame, x1, y1, x2, y2, color, t=1):
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, t)

def lbl(frame, txt, x, y, col=None, sc=0.40, tk=1):
    cv2.putText(frame, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                sc, col or C_TEXT_SEC, tk, cv2.LINE_AA)

def val(frame, txt, x, y, col=None, sc=0.60, tk=2):
    cv2.putText(frame, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                sc, col or C_TEXT_PRI, tk, cv2.LINE_AA)

def pill_bar(frame, x, y, w, h, v, mx, col):
    pct = min(v/mx, 1.0)
    cv2.rectangle(frame, (x,y), (x+w,y+h), C_BORDER, -1)
    fw = int(w*pct)
    if fw > 0:
        cv2.rectangle(frame, (x,y), (x+fw,y+h), col, -1)
        if fw > 3:
            glow = tuple(min(c+50,255) for c in col)
            cv2.rectangle(frame,(x+fw-2,y),(x+fw,y+h),glow,-1)

def badge(frame, x, y, w, h, txt, col):
    fill_alpha(frame, x, y, x+w, y+h, col, alpha=0.20)
    border(frame, x, y, x+w, y+h, col, 1)
    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, txt, (x+(w-tw)//2, y+(h+th)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

def corner_box(frame, x, y, w, h, col, s=12, t=2):
    for (px,py),(dx,dy) in zip(
        [(x,y),(x+w,y),(x,y+h),(x+w,y+h)],
        [(1,1),(-1,1),(1,-1),(-1,-1)]
    ):
        cv2.line(frame,(px,py),(px+dx*s,py),col,t,cv2.LINE_AA)
        cv2.line(frame,(px,py),(px,py+dy*s),col,t,cv2.LINE_AA)


# ──────────────────────────────────────────────
# DETECTORS
# ──────────────────────────────────────────────
class AgitationDetector:
    def __init__(self):
        self.history  = deque()
        self.prev_lms = None

    def update(self, lms):
        now = time.time()
        if self.prev_lms is not None:
            diffs = [np.sqrt((lms[i].x-self.prev_lms[i].x)**2+
                             (lms[i].y-self.prev_lms[i].y)**2)
                     for i in BODY_POINTS]
            self.history.append((now, float(np.mean(diffs))))
        self.prev_lms = lms
        while self.history and now-self.history[0][0] > AGITATION_WINDOW_SEC:
            self.history.popleft()
        if len(self.history) < 5:
            return 0.0, "READING"
        avg = float(np.mean([m for _,m in self.history]))
        if avg >= AGITATION_HIGH:   return avg, "HIGH"
        elif avg >= AGITATION_MILD: return avg, "MILD"
        return avg, "CALM"


class PostureDetector:
    def __init__(self):
        self.cal     = []
        self.ready   = False
        self.base_ny = None
        self.base_t  = None
        self.posture = "READING"

    def _tilt(self, lms):
        dy = abs(lms[11].y - lms[12].y)
        dx = abs(lms[11].x - lms[12].x)
        return dy/dx if dx > 0.01 else 0.0

    def update(self, lms):
        ny = lms[0].y
        t  = self._tilt(lms)
        if not self.ready:
            self.cal.append((ny, t))
            if len(self.cal) >= 25:
                self.base_ny = np.mean([f[0] for f in self.cal])
                self.base_t  = np.mean([f[1] for f in self.cal])
                self.ready   = True
                print("[MediSense] Posture calibrated.")
            return "CALIBRATING", 0.0
        nc   = self.base_ny - ny
        tc   = abs(t - self.base_t)
        prev = self.posture
        if nc > SITTING_UP_THRESHOLD:  self.posture = "SITTING UP"
        elif tc > ROLLING_THRESHOLD:   self.posture = "ROLLING"
        else:                          self.posture = "LYING"
        if self.posture != prev and self.posture != "LYING":
            log_event("POSTURE CHANGE", f"Patient {self.posture}", "WARNING")
        return self.posture, tc


class StillnessDetector:
    def __init__(self):
        self.since   = None
        self.prev    = None
        self.alerted = False
        self.skip    = 40

    def update(self, lms):
        now = time.time()
        if self.skip > 0:
            self.skip -= 1
            self.prev  = lms
            return 0.0, "READING"
        if self.prev is not None:
            diffs = [np.sqrt((lms[i].x-self.prev[i].x)**2+
                             (lms[i].y-self.prev[i].y)**2)
                     for i in BODY_POINTS]
            mv = float(np.mean(diffs))
            if mv < MOVEMENT_THRESHOLD:
                if self.since is None: self.since = now
                dur = now - self.since
            else:
                self.since   = None
                self.alerted = False
                dur          = 0.0
        else:
            dur = 0.0
        self.prev = lms
        if dur >= UNCONSCIOUS_SEC:
            if not self.alerted:
                self.alerted = True
                log_event("UNCONSCIOUS", f"Still {dur:.0f}s", "CRITICAL")
            return dur, "UNCONSCIOUS"
        elif dur >= UNCONSCIOUS_SEC*0.5: return dur, "VERY STILL"
        elif dur > 1.5:                  return dur, "STILL"
        return 0.0, "ACTIVE"


def check_boundary(lms):
    cx = float(np.mean([lms[i].x for i in [11,12,23,24]]))
    cy = float(np.mean([lms[i].y for i in [11,12,23,24]]))
    if cx < BED_MARGIN:          return "LEFT EDGE",  cx, cy
    if cx > 1.0 - BED_MARGIN:   return "RIGHT EDGE", cx, cy
    return "CENTER", cx, cy


def get_state(agi_st, posture, still_st, boundary_st):
    if still_st   == "UNCONSCIOUS":    return "CRITICAL", "UNCONSCIOUS — No movement"
    if agi_st     == "HIGH":           return "CRITICAL", "HIGH AGITATION — Pain signal"
    if boundary_st != "CENTER":        return "CRITICAL", f"BED EDGE — {boundary_st}"
    if posture in ["SITTING UP","ROLLING"]: return "WARNING", f"POSTURE — {posture}"
    if agi_st     == "MILD":           return "WARNING",  "RESTLESS — Discomfort"
    if still_st   == "VERY STILL":     return "WARNING",  "VERY STILL — Monitoring"
    return "NORMAL", "Patient stable"


# ──────────────────────────────────────────────
# RENDER PROFESSIONAL UI
# ──────────────────────────────────────────────
def render(frame, agi_v, agi_st, posture,
           still_dur, still_st, boundary_st,
           cx, cy, state, msg, face_vis):

    h, w = frame.shape[:2]
    sc   = (C_OK if state=="NORMAL" else C_WARN if state=="WARNING" else C_CRIT)

    # ── LEFT PANEL ──
    PX,PY,PW = 10,10,264
    PH       = h - 20
    fill_alpha(frame, PX, PY, PX+PW, PY+PH, C_PANEL, 0.88)
    border(frame, PX, PY, PX+PW, PY+PH, C_BORDER)

    # Status header
    fill_alpha(frame, PX, PY, PX+PW, PY+42, sc, 0.15)
    border(frame, PX, PY, PX+PW, PY+42, sc)
    cv2.circle(frame, (PX+18,PY+21), 6, sc, -1, cv2.LINE_AA)
    cv2.putText(frame, state, (PX+30,PY+27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, sc, 2, cv2.LINE_AA)

    # ── AGITATION ──
    sy = PY+55
    lbl(frame, "AGITATION", PX+10, sy)
    ac = C_CRIT if agi_st=="HIGH" else C_WARN if agi_st=="MILD" else C_OK
    pill_bar(frame, PX+10, sy+6, PW-20, 5, agi_v, AGITATION_HIGH, ac)
    badge(frame, PX+10, sy+18, 80, 20, agi_st, ac)

    # ── POSTURE ──
    sy = PY+113
    lbl(frame, "POSTURE", PX+10, sy)
    pc = (C_WARN if posture in ["SITTING UP","ROLLING","CALIBRATING"]
          else C_TEXT_SEC if posture=="READING" else C_OK)
    val(frame, posture, PX+10, sy+20, pc, 0.55, 2)

    # ── MOVEMENT ──
    sy = PY+175
    lbl(frame, "MOVEMENT", PX+10, sy)
    mc = (C_CRIT if still_st=="UNCONSCIOUS" else
          C_WARN if still_st in ["VERY STILL","STILL"] else
          C_TEXT_SEC if still_st=="READING" else C_OK)
    if still_dur > 0:
        pill_bar(frame, PX+10, sy+6, PW-20, 5, still_dur, UNCONSCIOUS_SEC, mc)
        val(frame, still_st, PX+10, sy+24, mc, 0.50, 1)
        lbl(frame, f"{still_dur:.0f}s  of  {UNCONSCIOUS_SEC:.0f}s", PX+10, sy+42, mc)
    else:
        val(frame, still_st, PX+10, sy+20, mc, 0.52, 1)

    # ── BED POSITION ──
    sy = PY+255
    lbl(frame, "BED POSITION", PX+10, sy)
    bc = C_CRIT if boundary_st != "CENTER" else C_OK
    badge(frame, PX+10, sy+8, 120, 22, boundary_st, bc)

    # ── DETECTION MODE ──
    sy = PY+312
    lbl(frame, "DETECTION MODE", PX+10, sy)
    fc   = C_OK if face_vis else C_WARN
    ftxt = "FACE + BODY" if face_vis else "BODY ONLY"
    val(frame, ftxt, PX+10, sy+18, fc, 0.50, 1)

    # Divider
    cv2.line(frame, (PX+10, PY+348), (PX+PW-10, PY+348), C_BORDER, 1)

    # ── ALERT LOG ──
    sy = PY+358
    lbl(frame, "RECENT ALERTS", PX+10, sy)
    recent = list(reversed(alert_log[-5:]))
    for i, e in enumerate(recent):
        ey = sy + 16 + i*38
        if ey + 32 > PY+PH-8: break
        ec = C_CRIT if e["severity"]=="CRITICAL" else C_WARN
        fill_alpha(frame, PX+10, ey, PX+PW-10, ey+30, ec, 0.08)
        border(frame, PX+10, ey, PX+PW-10, ey+30, ec, 1)
        cv2.circle(frame, (PX+20, ey+10), 3, ec, -1, cv2.LINE_AA)
        lbl(frame, e["time"], PX+28, ey+13, C_TEXT_SEC, 0.36)
        lbl(frame, e["event"][:22], PX+14, ey+26, ec, 0.37)

    # ── TOP RIGHT CLOCK ──
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    fill_alpha(frame, w-185, 10, w-10, 58, C_PANEL, 0.88)
    border(frame, w-185, 10, w-10, 58, C_BORDER)
    lbl(frame, "MEDISENSE  v1.0", w-178, 28, C_ACCENT, 0.38)
    val(frame, ts, w-178, 50, C_TEXT_PRI, 0.52, 1)

    # ── BOTTOM ALERT BANNER ──
    if state != "NORMAL":
        bx1, by1 = PX+PW+8, h-58
        bx2, by2 = w-10, h-10
        fill_alpha(frame, bx1, by1, bx2, by2, sc, 0.12)
        border(frame, bx1, by1, bx2, by2, sc)
        cv2.rectangle(frame, (bx1, by1), (bx1+5, by2), sc, -1)
        lbl(frame, "NURSE ALERT", bx1+14, by1+18, sc, 0.42)
        val(frame, msg, bx1+14, by1+40, C_TEXT_PRI, 0.52, 1)

    # ── BED BOUNDARY DASHED LINES ──
    lx = int(w * BED_MARGIN)
    rx = int(w * (1.0-BED_MARGIN))
    for yp in range(0, h, 12):
        cv2.line(frame,(lx,yp),(lx,min(yp+7,h)),C_BED_LINE,1)
        cv2.line(frame,(rx,yp),(rx,min(yp+7,h)),C_BED_LINE,1)
    lbl(frame,"BED EDGE",lx+4,22,C_BED_LINE,0.36)
    lbl(frame,"BED EDGE",rx+4,22,C_BED_LINE,0.36)

    # ── BODY CENTER CROSSHAIR ──
    if 0.01 < cx < 0.99 and 0.01 < cy < 0.99:
        px,py_ = int(cx*w), int(cy*h)
        bc2 = C_CRIT if boundary_st != "CENTER" else C_ACCENT
        cv2.circle(frame,(px,py_),9,bc2,1,cv2.LINE_AA)
        cv2.circle(frame,(px,py_),2,bc2,-1,cv2.LINE_AA)
        for dx,dy_ in [(-14,0),(-8,0),(8,0),(14,0),(0,-14),(0,-8),(0,8),(0,14)]:
            if abs(dx)==14 or abs(dy_)==14:
                pass
            else:
                cv2.line(frame,(px+dx,py_+dy_),(px+(dx*14//8),py_+(dy_*14//8)),bc2,1)
        cv2.line(frame,(px-14,py_),(px-9,py_),bc2,1)
        cv2.line(frame,(px+9, py_),(px+14,py_),bc2,1)
        cv2.line(frame,(px,py_-14),(px,py_-9),bc2,1)
        cv2.line(frame,(px,py_+9), (px,py_+14),bc2,1)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    agi_det  = AgitationDetector()
    post_det = PostureDetector()
    stil_det = StillnessDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); return

    print("="*52)
    print("  MediSense v1.0  |  Phase 3  |  Body Signals")
    print("="*52)
    print("  Calibrating for ~2s — hold still...")
    print("  Q = quit\n")

    agi_v=0.0; agi_st="READING"; posture="READING"
    still_dur=0.0; still_st="READING"; boundary_st="CENTER"
    cx=0.5; cy=0.5; state="NORMAL"; msg="Patient stable"
    face_vis=False

    with mp_pose.Pose(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        model_complexity=1
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            h, w = frame.shape[:2]
            frame = cv2.convertScaleAbs(frame, alpha=0.82, beta=0)

            results  = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces    = face_cascade.detectMultiScale(gray,1.1,5,minSize=(50,50))
            face_vis = len(faces) > 0

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                agi_v,  agi_st    = agi_det.update(lms)
                posture, _        = post_det.update(lms)
                still_dur,still_st= stil_det.update(lms)
                boundary_st,cx,cy = check_boundary(lms)

                if boundary_st != "CENTER":
                    log_event("BED BOUNDARY", f"Near {boundary_st}", "CRITICAL")

                state, msg = get_state(agi_st, posture, still_st, boundary_st)

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=C_SKELETON, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=C_SKEL_CON, thickness=2)
                )

            render(frame, agi_v, agi_st, posture,
                   still_dur, still_st, boundary_st,
                   cx, cy, state, msg, face_vis)

            cv2.imshow("MediSense | Patient Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Session ended. Total alerts: {len(alert_log)}")
    for e in alert_log:
        print(f"  {e['time']} | {e['severity']:8} | {e['event']} — {e['detail']}")

if __name__ == "__main__":
    main()

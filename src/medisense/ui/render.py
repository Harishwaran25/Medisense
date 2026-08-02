"""Clinical dark-theme overlay: side panel, alert banner, bed-edge guides,
skeleton crosshair. Pure drawing code — no detection logic lives here."""
import datetime
import cv2
import mediapipe as mp
_POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

from medisense.config import Thresholds

# Colors (BGR)
C_PANEL = (28, 32, 42)
C_BORDER = (45, 55, 70)
C_OK = (80, 200, 120)
C_WARN = (40, 160, 240)
C_CRIT = (60, 60, 220)
C_TEXT_PRI = (220, 225, 235)
C_TEXT_SEC = (110, 120, 140)
C_ACCENT = (140, 210, 255)
C_BED_LINE = (60, 80, 120)

def draw_skeleton(frame, landmarks, min_vis=0.5, color=(0, 255, 140)):
    if not landmarks:
        return
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        if lm.visibility >= min_vis:
            pts.append((int(lm.x * w), int(lm.y * h)))
        else:
            pts.append(None)

    for a, b in _POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)

    for p in pts:
        if p:
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)
def fill_alpha(frame, x1, y1, x2, y2, color, alpha=0.85):
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def border(frame, x1, y1, x2, y2, color, t=1):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, t)


def lbl(frame, txt, x, y, col=None, sc=0.40, tk=1):
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col or C_TEXT_SEC, tk, cv2.LINE_AA)


def val(frame, txt, x, y, col=None, sc=0.60, tk=2):
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col or C_TEXT_PRI, tk, cv2.LINE_AA)


def pill_bar(frame, x, y, w, h, v, mx, col):
    pct = min(v / max(mx, 0.0001), 1.0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_BORDER, -1)
    fw = int(w * pct)
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x + fw, y + h), col, -1)


def badge(frame, x, y, w, h, txt, col):
    fill_alpha(frame, x, y, x + w, y + h, col, alpha=0.20)
    border(frame, x, y, x + w, y + h, col, 1)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, txt, (x + (w - tw) // 2, y + (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)


def render(frame, agi_v, agi_st, posture, still_dur, still_st,
           boundary_st, cx, cy, state, msg, face_vis,
           fallen, fall_ratio, emotion, pain_score, pain_state,
           alert_log, cfg: Thresholds, landmarks=None, pose_backend: str = "auto"):

    h, w = frame.shape[:2]
    sc = C_OK if state == "NORMAL" else C_WARN if state == "WARNING" else C_CRIT
    draw_skeleton(frame, landmarks)

    PX, PY, PW = 10, 10, 270
    PH = h - 20
    fill_alpha(frame, PX, PY, PX + PW, PY + PH, C_PANEL, 0.88)
    border(frame, PX, PY, PX + PW, PY + PH, C_BORDER)

    fill_alpha(frame, PX, PY, PX + PW, PY + 44, sc, 0.15)
    border(frame, PX, PY, PX + PW, PY + 44, sc)
    cv2.circle(frame, (PX + 18, PY + 22), 6, sc, -1, cv2.LINE_AA)
    cv2.putText(frame, state, (PX + 30, PY + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, sc, 2, cv2.LINE_AA)
    lbl(frame, msg[:32], PX + 10, PY + 42, sc, 0.33)

    sy = PY + 58
    lbl(frame, "FALL STATUS", PX + 10, sy)
    fc = C_CRIT if fallen else C_OK
    val(frame, "FALL DETECTED!" if fallen else "SAFE", PX + 10, sy + 18, fc, 0.52, 2)
    lbl(frame, f"ratio {fall_ratio:.3f}", PX + 10, sy + 36, C_TEXT_SEC, 0.34)

    sy = PY + 110
    lbl(frame, "AGITATION", PX + 10, sy)
    ac = C_CRIT if agi_st == "HIGH" else C_WARN if agi_st == "MILD" else C_OK
    pill_bar(frame, PX + 10, sy + 6, PW - 20, 5, agi_v, cfg.agitation_high, ac)
    badge(frame, PX + 10, sy + 18, 80, 20, agi_st, ac)

    sy = PY + 158
    lbl(frame, "POSTURE", PX + 10, sy)
    pc = (C_WARN if posture in ["SITTING UP", "ROLLING", "CALIBRATING"]
          else C_TEXT_SEC if posture == "READING" else C_OK)
    val(frame, posture, PX + 10, sy + 18, pc, 0.52, 2)

    sy = PY + 208
    lbl(frame, "MOVEMENT", PX + 10, sy)
    mc = (C_CRIT if still_st == "UNCONSCIOUS" else
          C_WARN if still_st in ["VERY STILL", "STILL"] else
          C_TEXT_SEC if still_st == "READING" else C_OK)
    if still_dur > 0:
        pill_bar(frame, PX + 10, sy + 6, PW - 20, 5, still_dur, cfg.unconscious_sec, mc)
        val(frame, still_st, PX + 10, sy + 24, mc, 0.46, 1)
        lbl(frame, f"{still_dur:.0f}s / {cfg.unconscious_sec:.0f}s", PX + 10, sy + 40, mc)
    else:
        val(frame, still_st, PX + 10, sy + 18, mc, 0.50, 1)

    sy = PY + 272
    lbl(frame, "PAIN SIGNAL", PX + 10, sy)
    pnc = C_CRIT if pain_state == "CRITICAL" else C_WARN if pain_state == "WARNING" else C_OK
    pill_bar(frame, PX + 10, sy + 6, PW - 20, 5, pain_score, 1.0, pnc)
    pain_label = "OFFLINE" if pain_state == "OFFLINE" else f"{emotion.upper()}  {pain_score:.0%}"
    lbl(frame, pain_label, PX + 10, sy + 22, pnc, 0.38)

    sy = PY + 316
    lbl(frame, "BED POSITION", PX + 10, sy)
    bc = C_CRIT if boundary_st != "CENTER" else C_OK
    badge(frame, PX + 10, sy + 8, 120, 22, boundary_st, bc)

    sy = PY + 360
    lbl(frame, "POSE / FACE", PX + 10, sy)
    mode = f"{pose_backend.upper()}" + (" +FACE" if face_vis else "")
    val(frame, mode[:22], PX + 10, sy + 16,
        C_OK if face_vis else C_WARN, 0.42, 1)

    cv2.line(frame, (PX + 10, PY + 392), (PX + PW - 10, PY + 392), C_BORDER, 1)

    sy = PY + 402
    lbl(frame, "RECENT ALERTS", PX + 10, sy)
    for i, e in enumerate(list(reversed(alert_log[-4:]))):
        ey = sy + 14 + i * 36
        if ey + 30 > PY + PH - 8:
            break
        ec = C_CRIT if e["severity"] == "CRITICAL" else C_WARN
        fill_alpha(frame, PX + 10, ey, PX + PW - 10, ey + 28, ec, 0.08)
        border(frame, PX + 10, ey, PX + PW - 10, ey + 28, ec, 1)
        cv2.circle(frame, (PX + 20, ey + 9), 3, ec, -1, cv2.LINE_AA)
        lbl(frame, e["time"], PX + 28, ey + 12, C_TEXT_SEC, 0.34)
        lbl(frame, e["event"][:26], PX + 14, ey + 24, ec, 0.35)

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    fill_alpha(frame, w - 190, 10, w - 10, 60, C_PANEL, 0.88)
    border(frame, w - 190, 10, w - 10, 60, C_BORDER)
    lbl(frame, "MEDISENSE  v5.2", w - 183, 28, C_ACCENT, 0.38)
    val(frame, ts, w - 183, 52, C_TEXT_PRI, 0.52, 1)

    if state != "NORMAL":
        bx1, by1, bx2, by2 = PX + PW + 8, h - 58, w - 10, h - 10
        fill_alpha(frame, bx1, by1, bx2, by2, sc, 0.12)
        border(frame, bx1, by1, bx2, by2, sc)
        cv2.rectangle(frame, (bx1, by1), (bx1 + 5, by2), sc, -1)
        lbl(frame, "NURSE ALERT", bx1 + 14, by1 + 18, sc, 0.42)
        val(frame, msg, bx1 + 14, by1 + 40, C_TEXT_PRI, 0.52, 1)

    if fallen:
        ov = frame.copy()
        cv2.rectangle(ov, (0, h // 2 - 45), (w, h // 2 + 45), (0, 0, 160), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "FALL DETECTED — ALERTING RELATIVE", (w // 2 - 260, h // 2 + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    lx = int(w * cfg.bed_margin)
    rx = int(w * (1.0 - cfg.bed_margin))
    for yp in range(0, h, 12):
        cv2.line(frame, (lx, yp), (lx, min(yp + 7, h)), C_BED_LINE, 1)
        cv2.line(frame, (rx, yp), (rx, min(yp + 7, h)), C_BED_LINE, 1)
    lbl(frame, "BED EDGE", lx + 4, 22, C_BED_LINE, 0.36)
    lbl(frame, "BED EDGE", rx + 4, 22, C_BED_LINE, 0.36)

    if 0.01 < cx < 0.99 and 0.01 < cy < 0.99:
        px, py_ = int(cx * w), int(cy * h)
        bc2 = C_CRIT if boundary_st != "CENTER" else C_ACCENT
        cv2.circle(frame, (px, py_), 9, bc2, 1, cv2.LINE_AA)
        cv2.circle(frame, (px, py_), 2, bc2, -1, cv2.LINE_AA)
        cv2.line(frame, (px - 14, py_), (px - 9, py_), bc2, 1)
        cv2.line(frame, (px + 9, py_), (px + 14, py_), bc2, 1)
        cv2.line(frame, (px, py_ - 14), (px, py_ - 9), bc2, 1)
        cv2.line(frame, (px, py_ + 9), (px, py_ + 14), bc2, 1)

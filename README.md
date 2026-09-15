# MediSense

**Contactless AI-powered patient safety monitoring**

MediSense is a computer-vision based monitoring system for bedridden patients — no wearables, no sensors on the patient. A single camera feed is enough to detect falls, agitation, unconsciousness, pain, and bed-boundary violations in real time, with fault-tolerant alerting to staff.

---

## Problem statement 

Bedridden patients (elderly, post-surgical, ICU, or immobile patients) are at continuous risk of falls, unnoticed distress, agitation, or rolling off the bed — especially between nurse rounds. Existing options fall short:

- **Manual checks** — gaps in coverage between rounds
- **Wearable sensors** — uncomfortable, removable, unreliable on unconscious or confused patients
- **Basic CCTV** — requires a human to constantly watch the screen, no automated alerting

A few critical minutes of undetected distress can be the difference that matters.

## Solution

MediSense turns a passive camera feed into an active, self-verifying safety monitor. It combines YOLO-based pose estimation, MediaPipe landmark tracking, camera-based respiration estimation, and a facial-emotion transformer to continuously assess patient state — and only escalates to a human when it's genuinely confident something is wrong.

```
┌─────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│  Camera /   │ ──▶ │   Vision Pipeline    │ ──▶ │   7 Detector          │
│  file /     │     │  YOLO-pose + 33-pt   │     │   Modules              │
│  RTSP       │     │  MediaPipe landmarks │     │  fall / agitation /    │
└─────────────┘     │  + face tracking     │     │  posture / boundary /  │
                     └────────────────────┘     │  stillness /            │
                                                  │  respiration / pain    │
                                                  └──────────┬──────────┘
                                                             │
                     ┌───────────────────────────────────────┘
                     ▼
          ┌─────────────────────┐     ┌───────────────────────┐
          │  Smoothing + 3-stage  │ ──▶ │   Alert Manager          │
          │  confirmation with    │     │  audio + Telegram + SMS, │
          │  decay (anti-false-   │     │  automatic channel        │
          │  alarm)               │     │  failover                 │
          └─────────────────────┘     └───────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  SQLite event log    │ ──▶  `medisense-report` shift handover
          └─────────────────────┘
```

### Key design decisions

- **Contactless** — pose and face landmarks replace wearables; works even on unconscious patients
- **Sleep is not an emergency** — stillness alone never escalates. A motionless patient is only critical when respiration cannot be detected either, so overnight monitoring doesn't bury staff in false alarms
- **Says when it cannot tell** — if the chest is occluded, the room is too dark, or the stream has frozen, the system reports respiration as *unverified* rather than absent. "Can't measure" and "not breathing" are different answers and only one of them alarms
- **False-alarm suppression** — rolling-window smoothing, majority-vote label stabilization, and a 3-stage confirmation that decays rather than resetting, so a one-frame dropout doesn't discard progress on a real event
- **Signals are labelled by provenance** — the OpenCV expression fallback is capped at WARNING and can never raise a critical alert; every logged event records which pose backend and which expression path produced it
- **Expressions are only read from faces** — a classifier returns a confident-looking label for any image, so only true face detections are scored, side-lying faces are rotated upright first, and crops that are too small, too blurred, or too uncertain report `UNCERTAIN` or `NOT ASSESSED` instead of a number
- **Maintained baselines** — the fall detector rejects a calibration window the patient moved through and absorbs repositioning, without letting drift swallow a genuine slow slide
- **Measurable offline** — any video file or RTSP URL can be used as the source, replayed against a video clock so second-based thresholds mean the same thing as on a live feed
- **Fault-tolerant alerting** — Telegram with SMS and local-audio fallback, so alert delivery isn't a single point of failure

## Features

| Detector | Signal |
|---|---|
| **Fall** | Torso drop / lateral ejection / tumble vs a maintained lying baseline |
| **Agitation** | Rolling-window movement variance |
| **Posture** | Sitting up / rolling / lying classification |
| **Stillness** | How long the patient has been motionless (sleep-aware, never critical alone) |
| **Respiration** | Chest-region intensity oscillation in the plausible breathing band |
| **Pain** | Facial-emotion transformer on detected, upright-corrected faces only |
| **Bed boundary** | Left/right edge proximity, center-safe zone tracking |

Patient state resolves as:

| Condition | Reported as |
|---|---|
| Still, breathing detected | `ASLEEP` — normal |
| Still, no breathing detected | `NO RESPIRATION` — critical |
| Still, respiration unmeasurable | `PROLONGED STILL` — warning, after a long delay |

Additional capabilities:
- Low-light / night-vision frame enhancement (CLAHE on the luminance channel)
- YOLO ↔ MediaPipe pose backend with automatic failover
- Live skeleton overlay for visual verification
- Real-time clinical-style monitoring dashboard (OpenCV UI)

### The overlay

The panel shows the current verdict, the respiration reading, how long the
patient has been still, and three state chips — and nothing else. Raw
diagnostics go to `medisense.log`, not the screen, and readings the system
cannot stand behind are printed as words rather than numbers:

| Panel reads | Meaning |
|---|---|
| `15 BPM` | Respiration measured in the plausible breathing band |
| `UNVERIFIED` | Chest occluded, room too dark, or stream frozen — not an apnoea claim |
| `NOT DETECTED` | A full measurement window with no chest movement |
| `NOT ASSESSED` | No clear view of the face; nothing was scored |
| `UNCERTAIN` | A face was scored but the model's own confidence was below threshold |

Tune the expression gates with `MEDISENSE_EMOTION_MIN_CONFIDENCE`,
`MEDISENSE_EMOTION_MIN_FACE_PX`, and `MEDISENSE_EMOTION_MIN_SHARPNESS` if the
panel reports expressions you don't trust, or reads `NOT ASSESSED` too often.

## Tech Stack

Python · OpenCV · Ultralytics YOLO (pose) · MediaPipe · Hugging Face Transformers · PyTorch · SQLite · pytest

## Installation guide

From the repository root:

```bash
pip install -e .
```

## Usage

```bash
medisense
```

Runs the live monitoring dashboard against your configured source. Press `Q` to quit, or `R` to re-arm the fall baseline after the bed or camera has been moved.

### Choosing a source

`MEDISENSE_CAM_SOURCE` accepts a webcam index, a video file, or a stream URL:

```bash
MEDISENSE_CAM_SOURCE=0                             medisense   # default webcam
MEDISENSE_CAM_SOURCE=clips/night_roll.mp4          medisense   # recorded clip
MEDISENSE_CAM_SOURCE=rtsp://cam.ward2/stream       medisense   # IP camera
MEDISENSE_CAM_SOURCE=clips/sleep.mp4 MEDISENSE_LOOP_VIDEO=1 medisense
```

Files replay against a video clock derived from the container's own timestamps, so thresholds expressed in seconds behave the same during replay as on a live camera. That is what makes offline measurements comparable to live behaviour — see `docs/VALIDATION.md`.

See `src/medisense/config.py` for the full set of thresholds and environment variables.

### Shift handover report

```bash
medisense-report            # last 8 hours from medisense_events.db
medisense-report 12         # last 12 hours
medisense-report 8 /path/to/medisense_events.db
```

Uses Google Gemini if `GEMINI_API_KEY` is set, and a structured local synthesis otherwise.

## Testing

```bash
pytest tests/ -v
```

The suite covers detector behaviour without needing a camera or model weights, and includes an end-to-end run of the main loop against a generated video clip.

## Project Status

Portfolio project — actively developed, and **not a certified medical device**. Camera-based respiration estimation in particular is sensitive to lighting, bedding, camera angle and distance, and is unvalidated here; the pain score is a distress proxy, not a clinical pain scale. See `docs/VALIDATION.md` for the validation methodology and current status, and `PROJECT_HANDOFF.txt` for architecture handoff notes.

## License

MIT

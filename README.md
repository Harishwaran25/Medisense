# MediSense

**Contactless AI-powered patient safety monitoring**

MediSense is a computer-vision based monitoring system for bedridden patients — no wearables, no sensors on the patient. A single camera feed is enough to detect falls, agitation, unconsciousness, pain, and bed-boundary violations in real time, with fault-tolerant alerting to staff.

---

## Problem

Bedridden patients (elderly, post-surgical, ICU, or immobile patients) are at continuous risk of falls, unnoticed distress, agitation, or rolling off the bed — especially between nurse rounds. Existing options fall short:

- **Manual checks** — gaps in coverage between rounds
- **Wearable sensors** — uncomfortable, removable, unreliable on unconscious or confused patients
- **Basic CCTV** — requires a human to constantly watch the screen, no automated alerting

A few critical minutes of undetected distress can be the difference that matters.

## Solution

MediSense turns a passive camera feed into an active, self-verifying safety monitor. It combines YOLO-based pose estimation, MediaPipe landmark tracking, and a facial-emotion transformer to continuously assess patient state — and only escalates to a human when it's genuinely confident something is wrong.

```
┌─────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│   Camera    │ ──▶ │   Vision Pipeline    │ ──▶ │   6 Detector          │
│ (incl. low- │     │  YOLO-pose + 33-pt   │     │   Modules              │
│  light)     │     │  MediaPipe landmarks │     │  fall / agitation /    │
└─────────────┘     │  + face tracking     │     │  posture / boundary /  │
                     └────────────────────┘     │  stillness / pain      │
                                                  └──────────┬──────────┘
                                                             │
                     ┌───────────────────────────────────────┘
                     ▼
          ┌─────────────────────┐     ┌───────────────────────┐
          │  Smoothing + 3-stage  │ ──▶ │   Alert Manager          │
          │  confirmation state   │     │  audio + SMS fallback,   │
          │  machine (anti-false- │     │  automatic channel        │
          │  alarm)               │     │  failover                 │
          └─────────────────────┘     └───────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  SQLite event log +   │
          │  LLM shift summaries  │
          └─────────────────────┘
```

### Key design decisions

- **Contactless** — pose and face landmarks replace wearables; works even on unconscious patients
- **False-alarm suppression** — rolling-window signal smoothing, majority-vote label stabilization, and a 3-stage confirmation state machine before any alert fires
- **Non-blocking distress detection** — a Hugging Face facial-emotion transformer runs on a background thread so it adds a signal without slowing the real-time detection loop
- **Fault-tolerant alerting** — automatic connectivity detection with SMS-to-local-audio channel failover, so alert delivery isn't a single point of failure
- **Auditability** — every event logged to SQLite, with LLM-generated shift summaries for staff handoff instead of raw logs

## Features

| Detector | Signal |
|---|---|
| **Fall** | Torso-drop-toward-floor detection via pose ratio |
| **Agitation** | Rolling-window movement variance |
| **Posture** | Sitting up / rolling / lying classification |
| **Stillness** | Prolonged inactivity → unconsciousness risk |
| **Pain** | Facial-emotion transformer, background-thread inference |
| **Bed boundary** | Left/right edge proximity, center-safe zone tracking |

Additional capabilities:
- Low-light / night-vision frame enhancement
- YOLO ↔ MediaPipe pose backend with automatic failover
- Live skeleton overlay for visual verification
- Real-time clinical-style monitoring dashboard (OpenCV UI)

## Tech Stack

Python · OpenCV · Ultralytics YOLO (pose) · MediaPipe · Hugging Face Transformers · PyTorch · SQLite · pytest

## Installation

From the repository root:

```bash
pip install -e .
```

## Usage

```bash
medisense
```

Runs the live monitoring dashboard against your configured camera source. See `src/medisense/config.py` for environment-variable configuration (camera source, thresholds, pose backend, headless mode).

## Testing

```bash
pytest tests/ -v
```

## Project Status

Portfolio project — actively developed. See `docs/VALIDATION.md` for validation methodology and `PROJECT_HANDOFF.txt` for architecture handoff notes.

## License

MIT

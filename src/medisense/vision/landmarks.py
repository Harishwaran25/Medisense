"""
Unified landmark types shared by YOLO and MediaPipe backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Landmark:
    """Normalized [0, 1] pose landmark compatible with MediaPipe indices."""

    x: float
    y: float
    visibility: float = 1.0


NUM_LANDMARKS = 33

# COCO-17 (Ultralytics YOLO-pose) → MediaPipe-compatible indices.
COCO_TO_MP = {
    0: 0,   # nose
    1: 2,   # left eye
    2: 5,   # right eye
    3: 7,   # left ear
    4: 8,   # right ear
    5: 11,  # left shoulder
    6: 12,  # right shoulder
    7: 13,  # left elbow
    8: 14,  # right elbow
    9: 15,  # left wrist
    10: 16, # right wrist
    11: 23, # left hip
    12: 24, # right hip
    13: 25, # left knee
    14: 26, # right knee
    15: 27, # left ankle
    16: 28, # right ankle
}


def empty_landmarks() -> list[Landmark]:
    return [Landmark(0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS)]


def snapshot_landmarks(landmarks: Sequence) -> list[Landmark]:
    """Deep-copy x/y/visibility so MediaPipe object reuse cannot alias frames."""
    out: list[Landmark] = []
    for lm in landmarks:
        try:
            out.append(
                Landmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    visibility=float(getattr(lm, "visibility", 1.0) or 0.0),
                )
            )
        except (AttributeError, TypeError, ValueError):
            out.append(Landmark(0.0, 0.0, 0.0))
    # Pad / trim to expected length.
    if len(out) < NUM_LANDMARKS:
        out.extend(empty_landmarks()[len(out):])
    return out[:NUM_LANDMARKS]


def face_center(landmarks: Sequence, min_vis: float = 0.3) -> tuple[float, float] | None:
    """Mean of visible face/head landmarks, or None if insufficient signal."""
    idxs = [0, 2, 5, 7, 8]
    pts = []
    try:
        for i in idxs:
            lm = landmarks[i]
            if float(getattr(lm, "visibility", 0.0) or 0.0) >= min_vis:
                pts.append((float(lm.x), float(lm.y)))
    except (IndexError, TypeError, ValueError, AttributeError):
        return None
    if len(pts) < 2:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def landmarks_from_coco(
    kpts_xy: Iterable,
    kpts_conf: Iterable | None,
    frame_w: int,
    frame_h: int,
    conf_threshold: float = 0.25,
) -> list[Landmark]:
    """
    Convert YOLO COCO-17 keypoints (pixel coords) into a 33-slot MediaPipe-like list.
    Missing/low-confidence points get visibility 0.
    """
    lms = empty_landmarks()
    if frame_w <= 0 or frame_h <= 0:
        return lms

    xy_list = list(kpts_xy)
    confs = list(kpts_conf) if kpts_conf is not None else [1.0] * len(xy_list)

    for i, xy in enumerate(xy_list):
        mp_idx = COCO_TO_MP.get(i)
        if mp_idx is None:
            continue
        try:
            x_px, y_px = float(xy[0]), float(xy[1])
            c = float(confs[i]) if i < len(confs) else 0.0
        except (TypeError, ValueError, IndexError):
            continue
        if c < conf_threshold or not (x_px > 0 or y_px > 0):
            # YOLO often uses (0,0) for missing keypoints.
            if x_px == 0.0 and y_px == 0.0:
                continue
            if c < conf_threshold:
                continue
        lms[mp_idx] = Landmark(
            x=max(0.0, min(1.0, x_px / frame_w)),
            y=max(0.0, min(1.0, y_px / frame_h)),
            visibility=max(0.0, min(1.0, c)),
        )

    nose = lms[0]
    if nose.visibility > conf_threshold:
        for idx in (1, 3, 4):
            if lms[idx].visibility <= 0:
                lms[idx] = Landmark(nose.x, nose.y, nose.visibility * 0.5)
    return lms

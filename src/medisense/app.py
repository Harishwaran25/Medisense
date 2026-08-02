"""
MediSense — main application loop.

Run:
    python -m medisense
    medisense
"""
from __future__ import annotations

import logging
import signal
import sys

import cv2
import numpy as np

from medisense.config import Thresholds, setup_logging
from medisense.health import run_healthcheck
from medisense.alerting.alert_manager import AlertManager
from medisense.detectors.fall import FallDetector
from medisense.detectors.agitation import AgitationDetector
from medisense.detectors.posture import PostureDetector
from medisense.detectors.stillness import StillnessDetector
from medisense.detectors.boundary import check_boundary
from medisense.detectors.pain import PainDetector
from medisense.smoothing import LabelSmoother
from medisense.state import get_overall_state
from medisense.ui.render import render
from medisense.vision.pose import PoseEstimator
from medisense.vision.face import FaceFinder
from medisense.vision.landmarks import face_center
from medisense.vision.nightvision import NightVisionEnhancer

logger = logging.getLogger("medisense.app")


class _GracefulExit(Exception):
    pass


def _install_signal_handlers(stop_flag: dict):
    def _handler(signum, _frame):
        logger.info("Signal %s received — shutting down.", signum)
        stop_flag["stop"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def main() -> int:
    setup_logging()
    cfg = Thresholds()

    problems = cfg.validate()
    if problems:
        for p in problems:
            logger.error("Config error: %s", p)
        return 2

    logger.info("MediSense starting up.")

    try:
        report = run_healthcheck(cfg, warm_pose=True)
        report.raise_if_fatal()
    except Exception as e:
        logger.critical("Startup health check failed: %s", e)
        return 1

    alert = None
    pain_det = None
    pose_est = None
    cap = None
    stop_flag = {"stop": False}
    _install_signal_handlers(stop_flag)

    try:
        alert = AlertManager(cfg)
        pain_det = PainDetector(cfg)
        agi_det = AgitationDetector(cfg)
        post_det = PostureDetector(cfg)
        stil_det = StillnessDetector(cfg)
        fall_det = FallDetector(cfg)
        boundary_smooth = LabelSmoother(10, initial="CENTER")
        pose_est = PoseEstimator(cfg)
        face_finder = FaceFinder()
        night_vision = NightVisionEnhancer()

        if not pose_est.ready:
            logger.critical("No pose backend ready after init.")
            return 1

        logger.info(
            "Pose backend=%s | expression mode pending | headless=%s",
            pose_est.backend_name,
            cfg.headless,
        )

        cap = cv2.VideoCapture(cfg.cam_source)
        if not cap.isOpened():
            logger.critical("Cannot open camera source %s.", cfg.cam_source)
            return 1

        # Improve capture stability where drivers support it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        agi_v, agi_st, posture = 0.0, "READING", "READING"
        still_dur, still_st, boundary_st = 0.0, "READING", "CENTER"
        cx, cy, state, msg = 0.5, 0.5, "NORMAL", "Patient stable"
        face_vis, fallen, fall_ratio = False, False, 1.0
        emotion, pain_score, pain_state = "neutral", 0.0, "NORMAL"
        pose_backend = pose_est.backend_name
        frame_count = 0
        consecutive_read_failures = 0
        consecutive_pose_misses = 0
        window_name = "MediSense | Patient Monitoring"

        while not stop_flag["stop"] and cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures == 1 or consecutive_read_failures % 30 == 0:
                    logger.warning(
                        "Camera frame read failed (%d in a row).",
                        consecutive_read_failures,
                    )
                if consecutive_read_failures >= cfg.camera_fail_alert_frames:
                    alert.trigger(
                        "CRITICAL",
                        "CAMERA OFFLINE",
                        "No frames received — monitoring degraded.",
                    )
                if consecutive_read_failures >= cfg.camera_fail_exit_frames:
                    logger.critical("Camera unresponsive too long — shutting down.")
                    break
                continue

            consecutive_read_failures = 0
            alert.resolve("CAMERA OFFLINE")

            try:
                if frame.ndim != 3 or frame.shape[0] < 64 or frame.shape[1] < 64:
                    logger.warning("Skipping undersized/invalid frame.")
                    continue

                frame = cv2.convertScaleAbs(frame, alpha=0.82, beta=0)
                frame, is_low_light, mean_lum = night_vision.process(frame)
                frame_count += 1

                pose = pose_est.process(frame)
                pose_backend = pose.backend or pose_est.backend_name
                lms = pose.landmarks
                raw_lms = pose.raw_landmarks
                logger.info("DEBUG lms: %s items, sample=%s", len(lms) if lms else 0, lms[0] if lms else None)

                face_vis, face_box, face_crop = face_finder.find(
                    frame, landmarks=lms, person_bbox=pose.person_bbox
                )

                if (
                    face_vis
                    and face_crop is not None
                    and frame_count % cfg.emotion_every_n_frames == 0
                ):
                    pain_det.update(face_crop)
                emotion, pain_score, pain_state = pain_det.get()
                # Never escalate pain while model is still loading.
                if pain_state == "LOADING":
                    pain_state = "NORMAL"

                if lms is not None:
                    consecutive_pose_misses = 0
                    alert.resolve("POSE LOST")

                    fallen, fall_ratio = fall_det.update(lms)
                    agi_v, agi_st = agi_det.update(lms)
                    posture, _ = post_det.update(lms)

                    fc = face_center(lms) if face_vis else None
                    f_cx = fc[0] if fc else None
                    f_cy = fc[1] if fc else None
                    still_dur, still_st = stil_det.update(lms, f_cx, f_cy)

                    raw_boundary, cx, cy = check_boundary(lms, cfg)
                    boundary_st = boundary_smooth.update(raw_boundary)

                    pose_est.draw(frame, pose)
                    if face_box is not None:
                        x, y, fw, fh = face_box
                        cv2.rectangle(
                            frame, (x, y), (x + fw, y + fh), (140, 210, 255), 1
                        )
                else:
                    consecutive_pose_misses += 1
                    if consecutive_pose_misses >= cfg.pose_miss_clear_frames:
                        fallen = False
                        still_st = "READING"
                        still_dur = 0.0
                        agi_st = "READING"
                        boundary_st = "CENTER"
                        posture = "READING"
                        for ev in (
                            "FALL DETECTED",
                            "UNCONSCIOUS",
                            "HIGH AGITATION",
                            "BED BOUNDARY",
                            "POSTURE CHANGE",
                        ):
                            alert.resolve(ev)
                        if consecutive_pose_misses == cfg.pose_miss_clear_frames:
                            alert.trigger(
                                "WARNING",
                                "POSE LOST",
                                "Patient pose not detected — check camera / occlusion.",
                            )

                state, msg = get_overall_state(
                    fallen, agi_st, posture, still_st, boundary_st, pain_state
                )

                if lms is not None:
                    if fallen:
                        alert.trigger(
                            "CRITICAL",
                            "FALL DETECTED",
                            "Patient may have fallen / left bed",
                        )
                    else:
                        alert.resolve("FALL DETECTED")

                    if still_st == "UNCONSCIOUS":
                        alert.trigger(
                            "CRITICAL",
                            "UNCONSCIOUS",
                            f"No movement for {still_dur:.0f}s",
                        )
                    else:
                        alert.resolve("UNCONSCIOUS")

                    if agi_st == "HIGH":
                        alert.trigger(
                            "CRITICAL",
                            "HIGH AGITATION",
                            "Possible pain or distress",
                        )
                    else:
                        alert.resolve("HIGH AGITATION")

                    if boundary_st != "CENTER":
                        alert.trigger(
                            "CRITICAL",
                            "BED BOUNDARY",
                            f"Patient near {boundary_st}",
                        )
                    else:
                        alert.resolve("BED BOUNDARY")

                    if posture in ("SITTING UP", "ROLLING"):
                        alert.trigger(
                            "WARNING",
                            "POSTURE CHANGE",
                            f"Patient is {posture}",
                        )
                    else:
                        alert.resolve("POSTURE CHANGE")

                if pain_state == "CRITICAL":
                    alert.trigger(
                        "CRITICAL",
                        "PAIN DETECTED",
                        f"Emotion: {emotion} — {pain_score:.0%}",
                    )
                else:
                    alert.resolve("PAIN DETECTED")

                if not cfg.headless:
                    render(
                        frame,
                        agi_v,
                        agi_st,
                        posture,
                        still_dur,
                        still_st,
                        boundary_st,
                        cx,
                        cy,
                        state,
                        msg,
                        face_vis,
                        fallen,
                        fall_ratio,
                        emotion,
                        pain_score,
                        pain_state,
                        alert.get_log(),
                        cfg,
                        landmarks=raw_lms,
                        pose_backend=pose_backend,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
            except Exception:
                logger.exception("Error processing frame %d — skipping.", frame_count)
                continue

    except _GracefulExit:
        pass
    except Exception:
        logger.exception("Fatal error in MediSense main loop.")
        return 1
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if not cfg.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if alert is not None:
            try:
                alert.resolve_all()
                alert.close()
                logger.info(
                    "Session ended. Total alerts fired: %d", len(alert.get_log())
                )
                for e in alert.get_log():
                    logger.info(
                        "%s | %-8s | %s — %s",
                        e["time"],
                        e["severity"],
                        e["event"],
                        e["detail"],
                    )
            except Exception:
                logger.exception("Error during alert shutdown.")
        if pain_det is not None:
            try:
                pain_det.close()
            except Exception:
                pass
        if pose_est is not None:
            try:
                pose_est.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

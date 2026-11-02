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
import time
from pathlib import Path

import cv2

from medisense.config import (
    EVENT_AGITATION,
    EVENT_BOUNDARY,
    EVENT_CAMERA_OFFLINE,
    EVENT_FALL,
    EVENT_NO_RESPIRATION,
    EVENT_PAIN,
    EVENT_POSE_LOST,
    EVENT_POSTURE,
    EVENT_STILLNESS_UNVERIFIED,
    PATIENT_EVENTS,
    Thresholds,
    setup_logging,
)
from medisense.health import run_healthcheck
from medisense.alerting.alert_manager import AlertManager
from medisense.detectors import stillness as still_states
from medisense.detectors.breathing import BreathingDetector
from medisense.detectors.fall import FallDetector
from medisense.detectors.agitation import AgitationDetector
from medisense.detectors.posture import PostureDetector
from medisense.detectors.stillness import StillnessDetector
from medisense.detectors.boundary import check_boundary
from medisense.detectors.pain import PainDetector
from medisense.smoothing import LabelSmoother
from medisense.state import get_overall_state
from medisense.ui.model import MonitorSnapshot
from medisense.ui.render import render
from medisense.vision.capture import FILE, frame_position_msec, open_capture
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
    recorder = None
    show_window = False
    stop_flag = {"stop": False}
    _install_signal_handlers(stop_flag)

    try:
        try:
            cap, source_kind, clock = open_capture(cfg.cam_source)
        except (FileNotFoundError, RuntimeError) as e:
            logger.critical("%s", e)
            return 1

        alert = AlertManager(cfg, clock=clock)
        pain_det = PainDetector(cfg, clock=clock)
        agi_det = AgitationDetector(cfg, clock=clock)
        post_det = PostureDetector(cfg)
        stil_det = StillnessDetector(cfg, clock=clock)
        breath_det = BreathingDetector(cfg)
        fall_det = FallDetector(cfg)
        boundary_smooth = LabelSmoother(10, initial="CENTER")
        pose_est = PoseEstimator(cfg)
        face_finder = FaceFinder()
        night_vision = NightVisionEnhancer()

        if not pose_est.ready:
            logger.critical("No pose backend ready after init.")
            return 1

        logger.info(
            "Pose backend=%s | source=%s (%s) | clock=%s | headless=%s",
            pose_est.backend_name,
            cfg.cam_source,
            source_kind,
            clock.kind,
            cfg.headless,
        )

        agi_v, agi_st, posture = 0.0, "READING", "READING"
        still_dur, still_st, boundary_st = 0.0, still_states.READING, "CENTER"
        cx, cy, state, msg = 0.5, 0.5, "NORMAL", "Patient stable"
        face_vis, fallen = False, False
        emotion, pain_score, pain_state = "neutral", 0.0, "NORMAL"
        breathing = None
        is_low_light = False
        pose_backend = pose_est.backend_name
        frame_count = 0
        consecutive_read_failures = 0
        consecutive_pose_misses = 0
        window_name = "MediSense | Patient Monitoring"
        expression_mode = "unavailable"
        recorder = None
        record_started_at = None
        show_window = not cfg.headless
        want_overlay = show_window or bool(cfg.record_path)

        if cfg.record_path:
            out_path = Path(cfg.record_path).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Live inference is far slower than the camera's advertised FPS.
            # Using the camera rate makes the saved demo play in fast-forward.
            # File sources keep container FPS so replay duration stays honest.
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if source_kind == FILE and src_fps > 1.0:
                fps = src_fps
            else:
                fps = 12.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            recorder = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not recorder.isOpened():
                logger.critical("Could not open video writer at %s", out_path)
                return 1
            record_started_at = time.monotonic()
            if cfg.record_seconds > 0:
                logger.info(
                    "Recording annotated demo to %s (%.1f fps, stop after %.0fs)",
                    out_path,
                    fps,
                    cfg.record_seconds,
                )
            else:
                logger.info("Recording annotated demo to %s (%.1f fps)", out_path, fps)

        while not stop_flag["stop"] and cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None:
                if source_kind == FILE:
                    # A file that runs out of frames has finished, not failed.
                    if cfg.loop_video and frame_count > 0:
                        logger.info("End of video — looping.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.info("End of video after %d frames.", frame_count)
                    break

                consecutive_read_failures += 1
                if consecutive_read_failures == 1 or consecutive_read_failures % 30 == 0:
                    logger.warning(
                        "Camera frame read failed (%d in a row).",
                        consecutive_read_failures,
                    )
                if consecutive_read_failures >= cfg.camera_fail_alert_frames:
                    alert.trigger(
                        "CRITICAL",
                        EVENT_CAMERA_OFFLINE,
                        "No frames received — monitoring degraded.",
                    )
                # Keep confirmation advancing while the feed is down.
                alert.tick()
                if consecutive_read_failures >= cfg.camera_fail_exit_frames:
                    logger.critical("Camera unresponsive too long — shutting down.")
                    break
                continue

            consecutive_read_failures = 0
            alert.resolve(EVENT_CAMERA_OFFLINE)

            try:
                if frame.ndim != 3 or frame.shape[0] < 64 or frame.shape[1] < 64:
                    logger.warning("Skipping undersized/invalid frame.")
                    continue

                now = clock.advance(frame_position_msec(cap) if source_kind == FILE else None)
                frame, is_low_light, _mean_lum = night_vision.process(frame)
                frame_count += 1

                pose = pose_est.process(frame)
                pose_backend = pose.backend or pose_est.backend_name
                lms = pose.landmarks
                raw_lms = pose.raw_landmarks

                face = face_finder.find(
                    frame, landmarks=lms, person_bbox=pose.person_bbox
                )
                face_vis = face.found

                # Only real cascade detections are scored. A landmark-derived
                # crop is a guess around the nose, and the classifier would
                # happily label a pillow as distressed.
                if (
                    face.is_detection
                    and face.crop is not None
                    and frame_count % cfg.emotion_every_n_frames == 0
                ):
                    pain_det.update(face.crop, is_detection=True)
                emotion, pain_score, pain_state = pain_det.get()
                expression_mode = pain_det.get_mode()
                # Never escalate pain while the model is still loading.
                if pain_state == "LOADING":
                    pain_state = "NORMAL"

                alert.set_context(
                    pose_backend=pose_backend,
                    distress_score=pain_score,
                    expression_mode=expression_mode,
                )

                if lms is not None:
                    consecutive_pose_misses = 0
                    alert.resolve(EVENT_POSE_LOST)

                    fallen, _fall_ratio = fall_det.update(lms)
                    agi_v, agi_st = agi_det.update(lms)
                    posture, _ = post_det.update(lms)
                    breathing = breath_det.update(frame, lms, now)

                    fc = face_center(lms) if face_vis else None
                    f_cx = fc[0] if fc else None
                    f_cy = fc[1] if fc else None
                    still_dur, still_st = stil_det.update(
                        lms, f_cx, f_cy, breathing_state=breathing.state
                    )

                    raw_boundary, cx, cy = check_boundary(lms, cfg)
                    boundary_st = boundary_smooth.update(raw_boundary)
                else:
                    consecutive_pose_misses += 1
                    if consecutive_pose_misses >= cfg.pose_miss_clear_frames:
                        fallen = False
                        still_st = still_states.READING
                        still_dur = 0.0
                        agi_st = "READING"
                        boundary_st = "CENTER"
                        posture = "READING"
                        breath_det.reset()
                        breathing = None
                        for ev in PATIENT_EVENTS:
                            alert.resolve(ev)
                        if consecutive_pose_misses == cfg.pose_miss_clear_frames:
                            alert.trigger(
                                "WARNING",
                                EVENT_POSE_LOST,
                                "Patient pose not detected — check camera / occlusion.",
                            )

                state, msg = get_overall_state(
                    fallen, agi_st, posture, still_st, boundary_st, pain_state
                )

                if lms is not None:
                    if fallen:
                        alert.trigger(
                            "CRITICAL",
                            EVENT_FALL,
                            "Patient may have fallen / left bed",
                        )
                    else:
                        alert.resolve(EVENT_FALL)

                    # Stillness alone is sleep. Absent respiration is not.
                    if still_st == still_states.NO_RESPIRATION:
                        alert.trigger(
                            "CRITICAL",
                            EVENT_NO_RESPIRATION,
                            f"Still {still_dur:.0f}s with no chest movement detected",
                        )
                    else:
                        alert.resolve(EVENT_NO_RESPIRATION)

                    if still_st == still_states.PROLONGED_STILL:
                        alert.trigger(
                            "WARNING",
                            EVENT_STILLNESS_UNVERIFIED,
                            f"Still {still_dur:.0f}s and respiration cannot be "
                            f"measured ({getattr(breathing, 'reason', 'unknown')})",
                        )
                    else:
                        alert.resolve(EVENT_STILLNESS_UNVERIFIED)

                    if agi_st == "HIGH":
                        alert.trigger(
                            "CRITICAL",
                            EVENT_AGITATION,
                            "Possible pain or distress",
                        )
                    else:
                        alert.resolve(EVENT_AGITATION)

                    if boundary_st != "CENTER":
                        alert.trigger(
                            "CRITICAL",
                            EVENT_BOUNDARY,
                            f"Patient near {boundary_st}",
                        )
                    else:
                        alert.resolve(EVENT_BOUNDARY)

                    if posture in ("SITTING UP", "ROLLING"):
                        alert.trigger(
                            "WARNING",
                            EVENT_POSTURE,
                            f"Patient is {posture}",
                        )
                    else:
                        alert.resolve(EVENT_POSTURE)

                if pain_state == "CRITICAL":
                    alert.trigger(
                        "CRITICAL",
                        EVENT_PAIN,
                        f"Emotion: {emotion} — {pain_score:.0%} ({expression_mode})",
                    )
                else:
                    alert.resolve(EVENT_PAIN)

                alert.tick()

                if want_overlay:
                    render(
                        frame,
                        MonitorSnapshot(
                            state=state,
                            message=msg,
                            respiration_state=getattr(breathing, "state", "NOT MEASURED"),
                            respiration_bpm=getattr(breathing, "bpm", 0.0),
                            respiration_reason=getattr(breathing, "reason", ""),
                            stillness_state=still_st,
                            stillness_sec=still_dur,
                            posture=posture,
                            agitation_state=agi_st,
                            boundary_state=boundary_st,
                            fallen=fallen,
                            emotion=emotion,
                            pain_score=pain_score,
                            pain_state=pain_state,
                            expression_mode=expression_mode,
                            face_source=face.source,
                            landmarks=raw_lms,
                            person_bbox=pose.person_bbox,
                            face_box=face.box,
                            patient_cx=cx,
                            patient_cy=cy,
                            pose_backend=pose_backend,
                            source_label=source_kind,
                            low_light=is_low_light,
                            alerts=alert.get_log(),
                        ),
                        cfg,
                    )

                if recorder is not None:
                    # Keep writer size stable even if a source renegotiates mid-run.
                    h, w = frame.shape[:2]
                    if (w, h) != (width, height):
                        frame = cv2.resize(frame, (width, height))
                    recorder.write(frame)
                    if (
                        cfg.record_seconds > 0
                        and record_started_at is not None
                        and (time.monotonic() - record_started_at) >= cfg.record_seconds
                    ):
                        logger.info(
                            "Reached MEDISENSE_RECORD_SECONDS=%.0f — stopping.",
                            cfg.record_seconds,
                        )
                        break

                if show_window:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        # Re-arm the fall baseline after the bed or camera moves.
                        logger.info("Operator requested fall recalibration.")
                        fall_det.recalibrate()
                        breath_det.reset()
            except Exception:
                logger.exception("Error processing frame %d — skipping.", frame_count)
                continue

    except _GracefulExit:
        pass
    except Exception:
        logger.exception("Fatal error in MediSense main loop.")
        return 1
    finally:
        if recorder is not None:
            try:
                recorder.release()
                logger.info("Demo recording saved.")
            except Exception:
                logger.exception("Error closing video recorder.")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if show_window:
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

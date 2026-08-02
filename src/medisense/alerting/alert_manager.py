"""
Core alert engine.

Channels:
  1. Local alarm sound (built-in Python wave/struct/math synth — no files needed)
  2. Telegram notification (primary)
  3. SMS fallback via Termux when Telegram fails or internet is down

Every event goes through a 3-stage confirmation before it actually fires
(watching -> stage1 warning -> stage2 fire), so a single noisy frame or a
brief 1-2 second blip never triggers a real alert. A cooldown then prevents
the same event from re-firing too often once it has.

Credentials are read from `medisense.config.secrets`, which loads them
from environment variables — never hardcoded here.
"""
from __future__ import annotations

import time
import threading
import datetime
import subprocess
import os
import wave
import struct
import math
import tempfile
import logging

import requests

from medisense.config import Thresholds, secrets
from medisense.reporting.db import EventLogger

logger = logging.getLogger("medisense.alerting")


def generate_alarm_wav(path: str):
    """Alternating high/low alarm tone, built from Python stdlib only."""
    sample_rate = 44100
    volume = 32000
    beep_dur = 0.3
    cycles = 6

    def make_tone(freq, dur):
        n = int(sample_rate * dur)
        fade = int(sample_rate * 0.01)
        samples = []
        for i in range(n):
            t = i / sample_rate
            env = min(i, n - i, fade) / fade
            samples.append(int(volume * env * math.sin(2 * math.pi * freq * t)))
        return samples

    pcm = []
    for _ in range(cycles):
        pcm += make_tone(1200, beep_dur)
        pcm += make_tone(800, beep_dur)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


class AlertManager:
    """
    Usage:
        alert = AlertManager(thresholds)
        alert.trigger("CRITICAL", "FALL DETECTED", "Patient may have fallen")
        alert.resolve("FALL DETECTED")
    """

    def __init__(self, cfg: Thresholds):
        self.cfg = cfg
        self._active = {}
        self._alerted = {}
        self._lock = threading.Lock()
        self._log = []
        self._stop = threading.Event()
        self._event_db = EventLogger()

        missing = secrets.missing()
        if missing:
            logger.warning(
                "Missing secrets: %s — Telegram alerts will be skipped until "
                "these are set (see .env.example).", ", ".join(missing)
            )

        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()
        logger.info(
            "AlertManager started. Stage1=%.1fs Stage2=%.1fs cooldown=%.1fs",
            cfg.stage1_sec, cfg.stage2_sec, cfg.alert_cooldown_sec,
        )

    # ── Public API ──────────────────────────────

    def trigger(self, severity: str, event: str, detail: str):
        """Call every frame a problem is currently detected."""
        now = time.time()
        with self._lock:
            if event not in self._active:
                self._active[event] = {
                    "since": now, "severity": severity, "detail": detail, "stage": 0,
                }
                logger.info("Watching: %s (%s)", event, severity)

    def resolve(self, event: str):
        """Call when the situation returns to normal."""
        with self._lock:
            self._active.pop(event, None)

    def resolve_all(self):
        with self._lock:
            self._active.clear()

    def get_log(self) -> list:
        with self._lock:
            return list(self._log)

    def close(self):
        self._stop.set()
        self.resolve_all()

    # ── 3-stage confirmation ────────────────────

    def _monitor_loop(self):
        while not self._stop.is_set():
            time.sleep(0.5)
            now = time.time()
            with self._lock:
                events = [
                    (event, dict(info)) for event, info in self._active.items()
                ]

            for event, info in events:
                duration = now - info["since"]

                if duration >= self.cfg.stage1_sec and info["stage"] == 0:
                    with self._lock:
                        if event in self._active and self._active[event]["stage"] == 0:
                            self._active[event]["stage"] = 1
                    logger.info("Stage 1: %s (%.1fs)", event, duration)

                elif duration >= self.cfg.stage2_sec and info["stage"] == 1:
                    with self._lock:
                        last = self._alerted.get(event, 0)
                        if now - last < self.cfg.alert_cooldown_sec:
                            continue
                        if event not in self._active or self._active[event]["stage"] != 1:
                            continue
                        self._active[event]["stage"] = 2
                        self._alerted[event] = now
                        detail = self._active[event]["detail"]
                        severity = self._active[event]["severity"]
                    threading.Thread(
                        target=self._fire_alert,
                        args=(severity, event, detail, duration),
                        daemon=True,
                    ).start()

    # ── Fire ─────────────────────────────────────

    def _fire_alert(self, severity: str, event: str, detail: str, duration: float):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        logger.warning(
            "ALERT FIRED: %s | severity=%s | detail=%s | duration=%.1fs",
            event, severity, detail, duration,
        )

        with self._lock:
            self._log.append({
                "time": ts, "event": event, "detail": detail,
                "severity": severity, "duration": duration,
            })
            self._event_db.log_event(
                event_type=event,
                severity=severity,
                detail=detail,
                duration_sec=duration,
            )

        t1 = threading.Thread(target=self._play_alarm, daemon=True)
        telegram_ok = {"ok": False}

        def _tg():
            telegram_ok["ok"] = self._send_telegram(severity, event, detail, duration)

        t2 = threading.Thread(target=_tg, daemon=True)
        t1.start()
        t2.start()
        t2.join(timeout=6.0)

        if not telegram_ok["ok"]:
            threading.Thread(
                target=self._send_sms, args=(severity, event, detail), daemon=True
            ).start()

    # ── Channel 1: local alarm sound ────────────

    def _play_alarm(self):
        path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path = tmp.name
            tmp.close()
            generate_alarm_wav(path)
            subprocess.run(
                ["aplay", "-q", path],
                timeout=15,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("Alarm sound fallback (aplay unavailable?): %s", e)
            for _ in range(10):
                print("\a", end="", flush=True)
                time.sleep(0.4)
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # ── Channel 2: Telegram ─────────────────────

    def _send_telegram(self, severity: str, event: str, detail: str, duration: float) -> bool:
        if not secrets.telegram_bot_token or not secrets.telegram_chat_id:
            logger.warning("Telegram not configured — skipping notification for %s", event)
            return False

        icon = "ALERT" if severity == "CRITICAL" else "WARN"
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        message = (
            f"*{icon} MEDISENSE {severity}*\n"
            f"Event: {event}\n"
            f"Detail: {detail}\n"
            f"Duration: {duration:.0f}s\n"
            f"Time: {ts}\n"
            f"Please check on the patient immediately."
        )
        url = f"https://api.telegram.org/bot{secrets.telegram_bot_token}/sendMessage"

        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": secrets.telegram_chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
            if resp.status_code == 200:
                logger.info("Telegram alert sent for %s", event)
                return True
            logger.error("Telegram send failed: %s %s", resp.status_code, resp.text)
            return False
        except requests.exceptions.ConnectionError:
            logger.error("No internet — Telegram send failed for %s", event)
            return False
        except Exception as e:
            logger.error("Telegram error: %s", e)
            return False

    # ── Channel 3: SMS fallback (Termux) ────────

    def _send_sms(self, severity: str, event: str, detail: str):
        if not secrets.relative_phone_number:
            logger.warning("No relative phone number configured — skipping SMS")
            return

        msg = f"MEDISENSE {severity}: {event} — {detail}"
        try:
            result = subprocess.run(
                ["termux-sms-send", "-n", secrets.relative_phone_number, msg],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                logger.info("SMS sent to configured number")
            else:
                logger.error("SMS failed: %s", result.stderr.decode())
        except FileNotFoundError:
            logger.error("Termux not found — install Termux + Termux:API to enable SMS fallback")
        except Exception as e:
            logger.error("SMS error: %s", e)

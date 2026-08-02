import time
from medisense.alerting.alert_manager import AlertManager
from medisense.config import Thresholds


def test_alert_stages_and_resolve():
    cfg = Thresholds(stage1_sec=0.2, stage2_sec=0.4, alert_cooldown_sec=60.0)
    alert = AlertManager(cfg)
    try:
        alert.trigger("CRITICAL", "TEST EVENT", "unit test")
        time.sleep(0.55)
        # Allow monitor loop to advance stages / fire
        time.sleep(0.6)
        log = alert.get_log()
        assert any(e["event"] == "TEST EVENT" for e in log)
        alert.resolve("TEST EVENT")
        with alert._lock:
            assert "TEST EVENT" not in alert._active
    finally:
        alert.close()

"""
Stillness escalation tests.

The headline case is `test_sleeping_patient_never_escalates`: the previous
design fired CRITICAL "UNCONSCIOUS" roughly every cooldown period all night,
because it treated twenty seconds of stillness as unconsciousness. Anything
that reintroduces that behaviour should fail here.
"""
from medisense.config import Thresholds
from medisense.detectors import breathing as resp
from medisense.detectors.stillness import (
    ACTIVE,
    ASLEEP,
    NO_RESPIRATION,
    PROLONGED_STILL,
    READING,
    STILL,
    StillnessDetector,
)
from conftest import FakeClock, lying_patient


def hold_still(det, clock, seconds, breathing_state, step=0.5, lms=None):
    """Feed identical landmarks for a span of time, returning (dur, state)."""
    lms = lms if lms is not None else lying_patient()
    dur, state = 0.0, READING
    for _ in range(int(seconds / step)):
        clock.advance(step)
        dur, state = det.update(lms, breathing_state=breathing_state)
    return dur, state


def test_sleeping_patient_never_escalates():
    """Hours of stillness with breathing present is sleep, not an emergency."""
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)

    seen = set()
    lms = lying_patient()
    for _ in range(int(3 * 3600 / 5)):     # three hours at 5s steps
        clock.advance(5.0)
        _, state = det.update(lms, breathing_state=resp.BREATHING)
        seen.add(state)

    assert NO_RESPIRATION not in seen
    assert PROLONGED_STILL not in seen
    assert seen <= {ACTIVE, STILL, ASLEEP}


def test_sustained_stillness_with_breathing_reports_asleep():
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)
    dur, state = hold_still(det, clock, cfg.asleep_sec + 20, resp.BREATHING)
    assert state == ASLEEP
    assert dur >= cfg.asleep_sec


def test_absent_respiration_escalates():
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)
    _, state = hold_still(det, clock, cfg.no_respiration_sec + cfg.still_sec + 20, resp.NO_BREATHING)
    assert state == NO_RESPIRATION


def test_absent_respiration_waits_for_confirmation():
    """One window of no detected breathing is not yet an emergency."""
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)
    _, state = hold_still(det, clock, cfg.still_sec + 2.0, resp.NO_BREATHING)
    assert state != NO_RESPIRATION


def test_unmeasurable_respiration_warns_and_does_not_claim_absence():
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)

    _, mid = hold_still(det, clock, cfg.asleep_sec, resp.UNVERIFIED)
    assert mid == STILL

    _, late = hold_still(det, clock, cfg.prolonged_stillness_sec, resp.UNVERIFIED)
    assert late == PROLONGED_STILL


def test_no_respiration_signal_at_all_is_treated_as_unmeasurable():
    """Passing None must never be read as 'not breathing'."""
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)
    _, state = hold_still(det, clock, cfg.no_respiration_sec + cfg.still_sec + 20, None)
    assert state != NO_RESPIRATION


def test_movement_resets_the_stillness_clock():
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)
    hold_still(det, clock, cfg.asleep_sec + 10, resp.NO_BREATHING)

    clock.advance(0.5)
    dur, _ = det.update(lying_patient(cx=0.7, cy=0.6), breathing_state=resp.NO_BREATHING)
    assert dur == 0.0


def test_body_out_of_view_reports_reading():
    cfg = Thresholds()
    clock = FakeClock()
    det = StillnessDetector(cfg, warmup_frames=0, clock=clock)
    hidden = lying_patient()
    for i in (11, 12, 23, 24):
        hidden[i] = type(hidden[i])(hidden[i].x, hidden[i].y, 0.0)
    dur, state = det.update(hidden, breathing_state=resp.BREATHING)
    assert (dur, state) == (0.0, READING)

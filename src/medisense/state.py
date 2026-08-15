"""Combines individual detector outputs into one overall patient status."""
from medisense.detectors import stillness as still


def get_overall_state(fallen, agi_st, posture, still_st, boundary_st, pain_state):
    """
    Returns (severity, message) for the highest-priority active condition.

    Critical order:  fall > absent respiration > high agitation > bed edge > pain
    Warning order:   unverifiable respiration > posture change > pain > restless
    Otherwise the patient is stable; sustained stillness with detected
    breathing reports as asleep, which is a normal state and not an alert.
    """
    if fallen:
        return "CRITICAL", "FALL DETECTED"
    if still_st == still.NO_RESPIRATION:
        return "CRITICAL", "NO RESPIRATION DETECTED"
    if agi_st == "HIGH":
        return "CRITICAL", "HIGH AGITATION"
    if boundary_st != "CENTER":
        return "CRITICAL", f"BED EDGE — {boundary_st}"
    if pain_state == "CRITICAL":
        return "CRITICAL", "PAIN DETECTED"
    if still_st == still.PROLONGED_STILL:
        return "WARNING", "STILL — RESPIRATION UNVERIFIED"
    if posture in ["SITTING UP", "ROLLING"]:
        return "WARNING", f"POSTURE — {posture}"
    if pain_state == "WARNING":
        return "WARNING", "PAIN WARNING"
    if agi_st == "MILD":
        return "WARNING", "RESTLESS"
    if still_st == still.ASLEEP:
        return "NORMAL", "Asleep — breathing detected"
    return "NORMAL", "Patient stable"

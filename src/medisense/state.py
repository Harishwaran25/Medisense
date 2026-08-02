"""Combines individual detector outputs into one overall patient status."""


def get_overall_state(fallen, agi_st, posture, still_st, boundary_st, pain_state):
    """Priority order: fall > unconscious > high agitation > bed edge > pain
    > posture change > pain warning > mild agitation > very still > normal."""
    if fallen:
        return "CRITICAL", "FALL DETECTED"
    if still_st == "UNCONSCIOUS":
        return "CRITICAL", "UNCONSCIOUS"
    if agi_st == "HIGH":
        return "CRITICAL", "HIGH AGITATION"
    if boundary_st != "CENTER":
        return "CRITICAL", f"BED EDGE — {boundary_st}"
    if pain_state == "CRITICAL":
        return "CRITICAL", "PAIN DETECTED"
    if posture in ["SITTING UP", "ROLLING"]:
        return "WARNING", f"POSTURE — {posture}"
    if pain_state == "WARNING":
        return "WARNING", "PAIN WARNING"
    if agi_st == "MILD":
        return "WARNING", "RESTLESS"
    if still_st == "VERY STILL":
        return "WARNING", "VERY STILL"
    return "NORMAL", "Patient stable"

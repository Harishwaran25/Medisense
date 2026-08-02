# Validation notes

This project has **not** been validated against a labeled clinical video set.
Thresholds in `Thresholds` are empirical starting points for a single camera
setup.

Recommended next step for credibility:

1. Record short clips: normal sleep, sit-up, roll, agitation, simulated
   off-bed / fall, face distress expressions.
2. Label frame ranges for each event class.
3. Measure precision / recall per detector and retune thresholds.

Until then, treat alerts as demo signals only.

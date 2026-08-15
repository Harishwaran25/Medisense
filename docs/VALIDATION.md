# Validation notes

This project has **not** been validated against a labeled clinical video set.
The thresholds in `Thresholds` are empirical starting points for a single
camera setup. Until measured numbers exist, treat alerts as demo signals.

## Why this file matters more than the feature list

Every threshold in `config.py` is currently an opinion. The structure of the
system, its failover behaviour and its packaging can all be verified by
reading the code; whether it detects the right thing at the right moment
cannot. Measuring that is what turns each threshold from an argument into a
number.

## What is now possible

`MEDISENSE_CAM_SOURCE` accepts a video file, and files are replayed against a
video clock built from the container's own presentation timestamps. This
matters more than it sounds: frames from a file arrive as fast as they decode,
so without it twenty seconds of footage can elapse in two seconds of wall time
and every duration threshold (stillness, respiration, alert confirmation)
silently means something different during replay than it does live. With the
video clock, a number measured offline is the same number that applies to a
live feed.

## Methodology

1. **Record clips** covering, at minimum:
   - normal sleep (long — at least 20 minutes, ideally a full night)
   - sitting up, rolling, being repositioned by staff
   - agitation / restlessness
   - simulated off-bed slide and simulated tumble
   - occluded chest (blanket pulled up), dark room, and a deliberately
     frozen/stalled stream
   - facial distress expressions

2. **Label frame ranges** per event class. Include the negative classes
   explicitly — "asleep and fine" is the class the system used to get wrong,
   and it will not appear in the results unless it is labelled.

3. **Replay and score.** Run each clip and compare fired alerts against the
   labels. Report per detector:
   - precision and recall
   - median latency from event onset to alert
   - **false alarms per hour of normal sleep** — the single number that
     decides whether staff keep the system switched on

4. **Score the three respiration outcomes separately.** `BREATHING`,
   `NO BREATHING` and `UNVERIFIED` are not one metric. Two failures matter
   most, and they are not symmetric:
   - reporting `NO BREATHING` for a breathing patient (false critical)
   - reporting `UNVERIFIED` so often that the channel is useless in practice

   A frozen stream must land in `UNVERIFIED`, never `NO BREATHING`.

5. **Retune, then re-measure** on held-out clips rather than the clips used
   for tuning.

## Known limitations to state alongside any results

- Camera-based respiration depends on lighting, bedding, camera angle,
  distance, and how much of the chest is visible. It is not a certified
  respiration monitor and there is no apnoea-detection claim here.
- The pain score is a distress proxy derived from facial-emotion labels, not
  PSPI or PAINAD. When the neural model is unavailable the OpenCV fallback is
  a contrast/edge heuristic with no established relationship to pain; it is
  capped at WARNING and recorded as `cv_fallback` in the event log so it can
  never be mistaken for a model output.
- Bed-boundary margins assume a roughly fixed camera position relative to the
  bed.
- The fall baseline absorbs slow repositioning. Drift stops once the deviation
  passes halfway to the alert threshold and only applies while the patient is
  stationary, but a sufficiently gradual movement will still be partly
  absorbed. Press `R` to re-arm the baseline after moving the bed or camera.

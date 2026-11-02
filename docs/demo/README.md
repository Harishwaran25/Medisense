# Demo assets

- `medisense_demo.mp4` — annotated webcam recording (skeleton + metrics panel)
- `thumbnail.jpg` — poster frame for README / LinkedIn

## Re-record

Stand or sit fully in frame, then:

```bash
cd /path/to/Medisense
MEDISENSE_CAM_SOURCE=0 \
MEDISENSE_RECORD_PATH=docs/demo/medisense_demo.mp4 \
MEDISENSE_RECORD_SECONDS=75 \
medisense
```

Suggested moves: still → sit up → fidget → lean toward the frame edge.

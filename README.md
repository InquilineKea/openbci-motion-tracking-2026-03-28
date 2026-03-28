# OpenBCI Motion Tracking

Standalone export of the OpenBCI and webcam tooling built on March 28, 2026.

Contents:
- Real-time Cyton serial EEG parser and analyzers
- Live matplotlib EEG/PSD/band-power display
- Webcam hand/object event tagger
- Webcam eye tracker and fixation metrics
- Eyes-closed comparison and hand-object EEG correlation scripts
- Cloudflare Worker for public live status, spectrum pages, and D1 archival

Key files:
- `openbci_cyton_serial_metrics.py`
- `openbci_cyton_realtime_metrics.py`
- `openbci_cyton_realtime_analyzer.py`
- `openbci_cyton_live_plot.py`
- `webcam_event_tagger.py`
- `webcam_eye_tracker.py`
- `eye_fixation_metrics.py`
- `eyes_closed_compare.py`
- `hand_object_eeg_correlation.py`
- `push_openbci_status_online.py`
- `OPENBCI_ONLINE.md`
- `openbci_status_worker/`

Notes:
- Runtime outputs, caches, previews, and generated status files are intentionally excluded.
- The worker code is configured for Cloudflare Workers + KV + D1.

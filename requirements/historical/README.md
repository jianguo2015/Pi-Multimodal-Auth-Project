# Historical dependency pins (do not use)

These two files are the *unmodified* requirement lists recovered from the
audited project. They are kept for provenance only.

| file | what it was | usable today? |
| --- | --- | --- |
| `requirements_pc.txt` | the PC training/export environment (unpinned) | superseded by `requirements/runtime.txt` + `requirements/research.txt` |
| `requirements_pi.txt` | the Raspberry Pi 4B "edge" environment (pinned) | **no** - see below |

## Why `requirements_pi.txt` cannot run this repository

```
onnxruntime==1.16.0
opencv-python-headless==4.8.1.78
sounddevice==0.4.6
psutil==5.9.5
numpy==1.24.3
PyYAML==6.0
```

* The INT8 graphs shipped in `weights/onnx/` contain `ConvInteger` nodes.
  onnxruntime cannot even **parse** them before 1.24.1, so importing the model
  with the pinned 1.16.0 fails at session creation. The supported pin is
  `onnxruntime==1.24.1` (see `requirements/runtime.txt`).
* `psutil` is only needed by `app_pi/hardware_monitor.py`.
* No Raspberry Pi runtime measurement exists for this project. The original
  report's "INT8 is 22x faster on the edge" figure was never reproduced, and on
  x86_64 the opposite holds: the INT8 graphs are about **25x slower** than
  fp32 (measured, see `docs/BENCHMARKS.md`).

The historical Pi entry point is preserved at `app_pi/pi_main.py` with a header
explaining its status. `docs/HISTORICAL_EDGE_DEPLOYMENT.md` has the full note.

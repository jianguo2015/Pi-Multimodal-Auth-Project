# Historical edge deployment (Raspberry Pi 4B)

**Status: historical. Nothing in this project was ever measured on a Raspberry
Pi, and no Pi deployment is supported.**

## What the original project claimed

`docs/PHASE0_AUDIT_REPORT.md` records the original report's claims. The two
that concerned the edge deployment were:

1. the INT8 quantised graphs are required for the Pi and are **"22× faster"**;
2. `app_pi/pi_main.py` is an end-to-end "Raspberry Pi multimodal authentication"
   application with a hardware performance report (latency, CPU, RAM,
   temperature).

Neither survives contact with the evidence, and neither was ever backed by a
measurement on a Pi.

## What is actually in the repository

| file | what it is |
| --- | --- |
| `app_pi/pi_main.py` | a Pi entry point with a **header explaining its status**; the code path is real but `dummy_capture()` feeds `np.random.randn` placeholders, i.e. no camera/microphone capture exists |
| `app_pi/hardware_monitor.py` | reads `/sys/firmware/devicetree/base/model` and `/sys/class/thermal/thermal_zone0/temp`; Pi-only, cannot run on Windows |
| `requirements/historical/requirements_pi.txt` | the original Pi pins, including `onnxruntime==1.16.0` |
| `scripts_pc/05_export_and_quantize.py` | the export/quantisation step that produced the INT8 graphs (now replaced by `scripts/export.py`) |

Phase 1 made exactly one functional change to `app_pi/pi_main.py`: the weight
directory was renamed (`weights/onnx_int8` → `weights/onnx`). The file remains
historical.

## Why the INT8 story does not hold up

* **`onnxruntime==1.16.0` cannot load the shipped graphs.** The INT8 graphs
  contain `ConvInteger` nodes, which require onnxruntime ≥ 1.24.1. The pinned Pi
  environment would fail at `InferenceSession(...)` with an unsupported-operator
  error, before any inference. (This was the Phase 0 "deployment blocker".)
* **INT8 is not faster here.** Measured on x86_64
  (`docs/BENCHMARKS.md`): 33.3 ms for the three INT8 graphs versus 1.27 ms for
  the fp32 ones — INT8 is ~26× *slower*. Dynamic quantisation mainly reduces
  size (1.9 MB → 0.6 MB).
* **Would a Pi change that?** Possibly: ARM Cortex-A72 has integer SIMD that
  x86 deployments of these kernels do not exploit the same way, and the
  original author's intent (smaller models, fewer FLOPs) is plausible. But
  "plausible" is not "measured", and this repository publishes no Pi number.
* **The latency report was not genuine.** Because capture is `randn`-based, the
  "end-to-end" latency printed by the original app excludes decoding,
  detection, MFCC extraction and the capture itself.

## If you want to try it on a Pi

You need, at minimum:

1. `onnxruntime>=1.24.1` for the Pi (arm64 wheels exist for recent releases) —
   do not use the historical pin;
2. a real capture path: OpenCV/V4L2 for the camera plus `sounddevice`/`arecord`
   for audio, feeding `multimodal_auth.preprocessing` (do not re-implement the
   preprocessing);
3. a threshold decision through `multimodal_auth.decision`, not a private copy;
4. a benchmark with `scripts/benchmark.py`-style methodology (warm-up, repeated
   runs, median **and** p95) so the result is comparable to
   `docs/BENCHMARKS.md`;
5. honesty in the write-up: report the device, the OS image, the onnxruntime
   version, thread counts, and the fact that the model was trained on one
   person.

Until then, the only defensible statement is: *"the models were exported to
ONNX and quantised with a Pi-oriented size budget; edge performance is
unmeasured."*

## Recommendations

* Keep `app_pi/` as provenance, not as a supported deliverable.
* Do not quote any Pi number. There is none.
* If an edge target becomes a goal, treat it as a new project with its own
  measurement plan (device, budget, acceptance criteria) — and expect the
  fp32 graphs to be the better starting point unless you measure otherwise.

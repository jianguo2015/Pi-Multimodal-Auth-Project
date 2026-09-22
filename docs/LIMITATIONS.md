# Limitations

This page is the authoritative, deliberately uncomfortable list. If you are
about to present, demo or build on this project, read it first.

## 1. The models were trained on a toy dataset

* **One** enrolled identity (`User_Me`): 5 face photos and 10 audio clips.
* Five "strangers", **not** real people: their face features were synthesised
  as distractor features by the original pipeline (`scripts_pc/00_setup_strangers.py`),
  and their audio is likewise synthetic. They are not a representative
  impostor population.
* Consequence: the published metrics (`genuine mean 0.9182`, `impostor max
  0.0511`) describe this toy set only. They are **not** an accuracy estimate,
  not a FAR/FRR estimate, and not evidence of biometric quality. A model that
  separates 1 identity from 5 synthetic ones can still be useless — or
  dangerous — on real people.

## 2. No evaluation protocol

There is no train/validation/test split, no cross-validation, no
false-accept/false-reject curve, no equal-error-rate computation and no score
calibration. The single threshold `0.85` was chosen by the original author on
this same tiny set, i.e. it is fitted to the data it is reported on.

## 3. No anti-spoofing, no liveness, no quality checks

* A printed photo, a screen replay or a recorded voice is indistinguishable from
  a live person here; nothing in the pipeline tests for liveness.
* There is no face-quality check (blur, occlusion, extreme pose, multiple
  faces): if detection finds several faces the **largest** one wins, silently.
* If Haar detection finds nothing, the pipeline **centre-crops** the image
  instead of failing. On a photo that merely *looks* like a face the score is
  still produced — it is a fallback, not a validation step.
* No minimum audio quality check beyond a 0.1 s length guard; silence,
  clipping and noise all pass.

## 4. No privacy or bias work

* No fairness/bias evaluation across skin tone, gender, age or accent.
* No consent, retention or anonymisation policy for enrolment data. The
  enrolment biometrics live in `data/` and are **not** part of the repository
  (see `docs/PRIVACY_AND_SECURITY.md`).
* Embeddings are 128-d float vectors, not revocable templates; they are
  stored/compared in a way that does not support key rotation.

## 5. Performance claims

* **No Raspberry Pi measurement exists.** The original report's "INT8 is 22×
  faster on the edge" was never measured. On the development host the opposite
  is true: the three INT8 graphs take ~32.1 ms against ~1.3 ms for fp32, i.e.
  INT8 is about **25× slower** (`docs/BENCHMARKS.md`). Dynamic INT8 quantisation
  mainly reduces model size here (≈1.9 MB → ≈0.6 MB); it does not speed up
  x86_64.
* The INT8 graphs need `onnxruntime>=1.24.1` (`ConvInteger`). The historical
  Pi pin `1.16.0` cannot even load them.
* Preprocessing dominates end-to-end latency: Haar detection on a full-size
  photo is far more expensive than all three networks together.
* Everything measured here is single-process, single-thread-pool, CPU only.

## 6. Scope and deployment

* No live capture: `app_pi/pi_main.py` uses `np.random.randn` placeholders and
  was never run on a Pi. There is no camera/microphone capture path in the
  supported code.
* The `--backend pytorch` path exposes the attention weights, the INT8 path
  cannot (the exported graph only returns the final score).
* `pytorch` and `onnx` differ by up to ~1e-2 on a score and ~0.045 on a single
  embedding dimension (quantisation error). Do not mix backends when comparing
  results.
* Threshold semantics are absolute: `score >= 0.85`. There is no per-user
  calibration and no rejection option ("I don't know"), so a borderline score
  is forced into ACCEPT or REJECT.

## 7. Engineering caveats of the environment

Windows + non-ASCII paths break several C/C++ libraries. The development
checkout happens to live under such a path (a Windows user directory with a
non-ASCII user name), which is how these were found:

* `cv2.imread`/`cv2.imwrite` cannot open non-ASCII paths. Phase 1 routes these
  through `cv2.imdecode`/`cv2.imencode` in `preprocessing/face.py`, which is why
  the shipped pipeline works in this location where the original scripts did
  not (the original `extract_face` returned `None` for **all 55** images).
* `onnxruntime.quantization` staging fails on non-ASCII paths because
  `onnx.shape_inference.infer_shapes_path` uses the narrow-character API.
  `scripts/export.py` detects this and stages in an ASCII-only scratch
  directory.
* Recommendation: clone into an ASCII-only path (e.g. `C:\src\multimodal-auth`)
  if you hit any other third-party path issue.

## 8. What is *not* claimed

* Not a production biometric system, not certified, not evaluated by any third
  party.
* Not a replacement for a real face/voice SDK; the CNNs are deliberately tiny
  (a Pi-oriented budget) and untuned.
* No guarantee that the reconstructed `face_extractor_weights.pth` matches the
  destroyed original byte-for-byte (it is functionally equivalent to 5.36e-07;
  see `docs/MODEL_ARTIFACT_MANIFEST.md`).
* No guarantee of reproducibility for *re-training*: the original runs were
  unseeded, so `scripts/train/*` can only reproduce the recipe, not the exact
  checkpoints.

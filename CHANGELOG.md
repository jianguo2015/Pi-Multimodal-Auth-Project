# Changelog

All notable changes to this project are documented here. This is a research
prototype; "release" means "this is the state that was audited, tested and
frozen", not "production ready".

## [1.0.0] - 2026-09-22

First public release. **The models are unchanged**: no retraining, no
architecture change, no threshold change, no numeric drift. The nine artifacts
under `weights/` are the audited originals, byte-identical
(`docs/MODEL_ARTIFACT_MANIFEST.md`, `tests/test_artifacts.py`).

### Engineering

* `src/multimodal_auth/` — installable package (config → preprocessing → models
  → fusion → decision → pipeline → CLI), importable as `multimodal_auth`
  without pulling in PyTorch.
* `configs/default.yaml` is the single source of truth; every value is validated
  on load, including cross-field checks, and the project root is discovered by
  walking up from the config file, so the repository can live anywhere.
* Scripts: `scripts/infer.py`, `scripts/benchmark.py`, `scripts/export.py`,
  `scripts/data/*`, `scripts/train/*` — all write **only** under `outputs/`.
* `scripts_pc/` (the original pipeline) is kept as provenance and fails closed
  unless `MULTIMODAL_AUTH_RUN_LEGACY=1` is set.
* Release packaging: `pyproject.toml` metadata, license expression, project URLs.

### Reproducibility

* Preprocessing is bit-identical to the audited originals (55/55 face tensors,
  35/35 voice tensors) and re-derived by the test suite from the raw media.
* The published score grids are reproducible: genuine mean 0.918214 (min
  0.894259, max 0.929255, accept 50/50) and impostor max 0.0511 (accept 0/625)
  over the expanded 625-pair grid.
* `docs/REPRODUCIBILITY.md` states exactly what can and cannot be re-derived,
  including the artifacts that are *not* reproducible (training runs, the
  original `.pth` bytes, BatchNorm statistics).

### Safety

* `src/multimodal_auth/safety.py` refuses writes to `weights/`, `data/`, `src/`,
  `configs/`, `tests/` and `app_pi/`; artifact-producing scripts fail closed on a
  non-empty output directory unless `--force` is given.
* `docs/MODEL_ARTIFACT_MANIFEST.md` pins size + SHA-256 of all nine shipped
  files; `tests/test_artifacts.py` fails if a single byte changes. This closes
  the Phase 0 incident class, in which a training script overwrote its own
  checkpoints.
* Invalid input raises instead of silently returning `None`.

### Testing

* 129 tests, split into a public tier (always runs) and a `private_data` tier
  (skipped automatically when the local biometric data is absent).
* Local full validation: **129 passed**. Fresh clone / CI:
  **125 passed, 4 skipped** — the four skips are named and explained in
  `docs/FRESH_CLONE_VALIDATION.md`.
* `.github/workflows/ci.yml` runs the public tier plus an end-to-end inference
  job on a clean Linux runner.

### Benchmark

* `docs/BENCHMARKS.md` now records the measurement honestly, including the
  environment, warm-up, runs, median and p95 — and records that on the tested
  x86_64 host the shipped **INT8 graphs are ~26× slower than fp32**
  (33.321 ms vs 1.267 ms for the three models). The original report's
  "INT8 is 22× faster" claim is not reproducible and has been withdrawn.
* No Raspberry Pi number is published, because none was ever measured.

### Fixed

* `opencv-python-headless` is pinned below 5.0: the 5.x wheels no longer ship
  `cv2/data/haarcascade_*.xml` (and expose no `cv2.CascadeClassifier`), so an
  unpinned install made every face input fail on a clean machine — reproduced
  during this audit, see `docs/RELEASE_AUDIT.md` §3.
* The declared Python floor was corrected from `>=3.10` to `>=3.11`: the pinned
  `onnxruntime==1.24.1` publishes cp311–cp314 wheels only, so the old claim was
  uninstallable (`docs/RELEASE_AUDIT.md` §3.2).
* `.gitignore` patterns are now anchored to the repository root. The unanchored
  `data/` pattern also matched `scripts/data/`, which silently kept the
  feature-building scripts (`scripts/data/01_build_voice_features.py`,
  `02_build_face_features.py`, `_common.py`) out of **every** commit although
  `docs/REPRODUCIBILITY.md` documents them (`docs/RELEASE_AUDIT.md` §3.3).
* `pytest -m "not private_data"` no longer fails during collection in an
  environment without the research extras: `tests/test_phase0_reference.py`
  defers its torch import and is reported as skipped instead of erroring.
* `scripts/infer.py --json` now reports repository-relative paths
  (`weights/onnx/...`) instead of machine-specific absolute paths, so a copied
  report cannot leak a local user name. CI asserts this.
* Documentation no longer contains machine-specific absolute paths (29 targeted
  substitutions across 11 files, assertion-checked).
* Licensing, attribution and privacy documents were added: `LICENSE`,
  `MODEL_NOTICE.md`, `THIRD_PARTY_NOTICES.md`, `docs/RELEASE_AUDIT.md`,
  `docs/MODEL_RELEASE.md`, `docs/FRESH_CLONE_VALIDATION.md`.

### Not included (deliberately)

* No live camera/microphone capture, no liveness or anti-spoofing, no web UI, no
  LLM/RAG/agent features, no Raspberry Pi support claim, no production accuracy
  claim, and no retrained or "improved" models. See `docs/LIMITATIONS.md`.

[1.0.0]: https://github.com/jianguo2015/multimodal-auth/releases/tag/v1.0.0

# Phase 1 change log

Phase 1 was an engineering pass: make the audited prototype safe, runnable,
testable and publishable **without changing the models**. This page lists every
change and the evidence for each.

## Explicitly out of scope

* **No model was retrained.** The `.pth`/`.onnx` artifacts are bit-identical to
  the audited ones (see the manifest).
* **No threshold or hyper-parameter was tuned.** `0.85` and every preprocessing
  constant are the audited values.
* **Nothing was pushed** to GitHub; the history is local (baseline commit
  `chore: freeze phase0 audited baseline` + one Phase 1 commit).
* **No LLM/RAG features were added**, and nothing depends on a network call.
* **No license was chosen** (see `docs/PRIVACY_AND_SECURITY.md` §5).
* **`app_pi/` was not made runnable** on a Pi; it stays historical.

## Added

| path | why |
| --- | --- |
| `src/multimodal_auth/` | the importable package: `errors`, `safety`, `config/`, `preprocessing/`, `models/`, `fusion/`, `decision/`, `pipeline/`, `reporting.py`, `cli.py`, `__main__.py` |
| `configs/default.yaml` | single, validated source of truth (threshold, shapes, preprocessing, runtime) |
| `pyproject.toml` | packaging metadata, dependencies, pytest config, `multimodal-auth` console script |
| `requirements/{runtime,dev,research}.txt` | split requirements, with the ORT pin and the reason for it |
| `requirements/historical/` | the original `requirements_pc.txt` / `requirements_pi.txt` plus an explanation of why the Pi pins cannot work |
| `scripts/infer.py` | one-shot verification CLI (ONNX-first, `--json`, exit codes) |
| `scripts/benchmark.py` | INT8-vs-fp32 and end-to-end latency; evidence in `outputs/benchmarks/` |
| `scripts/data/{_common,01_build_voice_features,02_build_face_features}.py` | non-destructive feature building with `--compare-to` bit-equality checking |
| `scripts/train/{_common,train_extractors,train_fusion}.py` | sandboxed re-training; `--report-shipped` re-verifies the Phase 0 numbers |
| `scripts/export.py` | PyTorch → ONNX fp32 → INT8 with numeric verification and an ASCII-safe quantisation scratch directory |
| `examples/make_examples.py` + generated samples | deterministic synthetic inputs so the repo runs right after cloning |
| `tests/` | 129 tests: config, preprocessing, fusion, decision, pipeline, artifacts, safety, CLI, and the `private_data` reference tier |
| `docs/MODEL_ARTIFACT_MANIFEST.md` | size + SHA-256 + provenance of all nine shipped artifacts (machine-read by tests) |
| `docs/{ARCHITECTURE,TESTING,REPRODUCIBILITY,BENCHMARKS,LIMITATIONS,PRIVACY_AND_SECURITY,HISTORICAL_EDGE_DEPLOYMENT,PHASE1_CHANGES}.md` | new documentation |
| `docs/benchmarks/20260922-development-host.{md,json}` | committed benchmark evidence |

## Moved (history preserved by `git mv` / rename detection)

| from | to | verification |
| --- | --- | --- |
| `models/face_extractor.py` | `src/multimodal_auth/models/face_extractor.py` | SHA-256 identical |
| `models/voice_extractor.py` | `src/multimodal_auth/models/voice_extractor.py` | SHA-256 identical |
| `models/attention_fusion.py` | `src/multimodal_auth/fusion/attention_fusion.py` | SHA-256 identical |
| `utils/vision_ops.py` | `src/multimodal_auth/preprocessing/face.py` | 55/55 bit-identical outputs |
| `utils/audio_ops.py` | `src/multimodal_auth/preprocessing/audio.py` | 35/35 bit-identical outputs |
| `weights/onnx_int8/` | `weights/onnx/` | hashes unchanged |
| `weights/pytorch_pth/` | `weights/pytorch/` | hashes unchanged |
| `requirements_pc.txt`, `requirements_pi.txt` | `requirements/historical/` | content unchanged |

## Modified

| path | change |
| --- | --- |
| `config/settings.yaml` | the two weight paths point at the renamed directories; header note added. Schema and values untouched (still read by `app_pi/` and the frozen scripts) |
| `app_pi/pi_main.py` | status header added; `weights/onnx_int8/` → `weights/onnx/` in the error message. No logic change |
| `scripts_pc/*.py` (7 files) | a fail-closed guard header prepended; each script now exits immediately unless `MULTIMODAL_AUTH_RUN_LEGACY=1`. Original bytes otherwise untouched |
| `.gitignore` | extended (private data, `_phase0_audit/`, `outputs/`, unrelated coursework) |

## Frozen (kept as evidence, not maintained)

* `scripts_pc/` — the original pipeline; superseded by `scripts/`, and disabled
  by the guard because its training step wrote back into `weights/`.
* `app_pi/` — the historical Pi entry point.
* `docs/PHASE0_*.md`, `docs/CORE_ARCHITECTURE.md`, `docs/PROJECT_CONTEXT.md`,
  `docs/CODEBASE_MAP.md`, `docs/OPEN_SOURCE_RISK.md` — the Phase 0 audit record.
  Deliberately not rewritten: `docs/CODEBASE_MAP.md` still describes the old
  layout, while `docs/ARCHITECTURE.md` documents the new one.

## Bugs found and fixed along the way

| # | bug | fix |
| --- | --- | --- |
| 1 | `utils/vision_ops.extract_face` returned `None` for **all 55** images in this checkout, because `cv2.imread` cannot open non-ASCII paths on Windows | `read_image_bgr` tries `imread` (ASCII paths only) and falls back to `imdecode` over bytes read by Python; `write_image_bgr` does the symmetric thing for output |
| 2 | `extract_mfcc` returned `None` on error, so failures silently became missing features | `preprocess_voice` raises `InvalidAudioError`, including for empty / undecodable / too-short clips |
| 3 | `librosa.feature.mfcc` on an empty waveform returns one padded frame, i.e. an all-zero tensor that would be fed to the model | explicit minimum-length guard inside `mfcc_features` |
| 4 | `scripts_pc/03` and `04` trained straight into `weights/pytorch_pth/` (the Phase 0 incident) | sandbox + guard: outputs only under `outputs/`, legacy scripts fail closed |
| 5 | `scripts_pc/05` repointed `TMP`/`TEMP` and never restored them | `scripts/export.py` restores `tempfile.tempdir` and the environment in a `finally` block |
| 6 | `onnxruntime.quantization` crashed with a confusing `FileNotFoundError` on non-ASCII paths (`infer_shapes_path` uses the ANSI API) | an ASCII-only scratch directory, chosen automatically with a writability probe |
| 7 | one threshold sat in a YAML file read by five scripts via relative paths, and the same numbers were hard-coded again in `utils/` and `models/` | one validated `configs/default.yaml` plus a tested `decision` package |
| 8 | `requirements_pi.txt` pins an onnxruntime that cannot load the shipped INT8 graphs | `requirements/runtime.txt` pins 1.24.1 and documents why; the old pins are kept as history |

## Evidence summary

| check | result |
| --- | --- |
| audited model code preserved byte-for-byte | 3/3 SHA-256 identical |
| face preprocessing vs original (ASCII-path probe) | 55/55 bit-identical |
| face preprocessing vs stored Phase 0 features | 55/55 bit-identical |
| voice preprocessing vs original and stored features | 35/35 bit-identical |
| genuine grid (PyTorch, stored features) | 0.918214 / 0.894259 / 0.929255, accept 50/50 |
| genuine grid (INT8 ONNX) | mean 0.9198, accept 50/50 |
| impostor grid (INT8 ONNX, 625 pairs) | max 0.0511, accept 0/625 |
| PyTorch vs INT8 per-pair score difference | ≤ 0.0028 |
| `weights/` hashes after running data / train / export / benchmark scripts | unchanged |
| test suite | 129 passed |

## Phase 2 correction to the counts above

"129 passed" is the *full* environment: runtime + test + research extras **and**
the private enrolment data present. The other environments give other (all
correct) numbers, and they are now stated everywhere instead of a single figure:

| environment | result |
| --- | --- |
| runtime + test requirements, no enrolment data (public clone) | 113 passed, 5 skipped, 4 deselected (`-m "not private_data"`) |
| runtime + test + research extras, no enrolment data | 125 passed, 4 skipped |
| everything + enrolment data | 129 passed |

The Phase 1 note "125 passed, 4 skipped on a fresh clone" therefore described the
*research-extras* environment. Phase 2 found three defects that the single number
was hiding and fixed them: `pytest -m "not private_data"` failed during collection
without torch, `scripts/data/*` was invisible to git, and the declared Python
floor was uninstallable. See `docs/RELEASE_AUDIT.md` §3 and
`docs/FRESH_CLONE_VALIDATION.md`.

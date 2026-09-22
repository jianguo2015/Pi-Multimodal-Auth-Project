# multimodal-auth — face + voice identity verification prototype

Two-modality identity verification: a face image and a short voice clip are
encoded by two small CNNs, fused by an **attention-gated** head, and compared
against a single calibrated threshold.

> ## Status: research prototype — not a security product
>
> This project was re-engineered in Phase 1 (structure, safety, tests,
> documentation). **The models themselves are unchanged** and were *not*
> retrained. They were trained on a **toy dataset** — one enrolled identity and
> five synthetic "strangers" — so every accuracy number below describes that
> toy setup only.
>
> Do not use this for real access control, and do not present it as a
> production biometric system. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## What it does

```
face image ──▶ FaceExtractor   ──▶ 128-d face embedding  ─┐
                                                          ├─▶ ModalAttentionFusion ─▶ score ─▶ ACCEPT/REJECT @0.85
voice clip ──▶ VoiceExtractor  ──▶ 128-d voice embedding ─┘
```

* face: Haar-cascade detection → 112×112 crop → `(x - 127.5) / 128`
* voice: 16 kHz mono → 40 MFCCs → crop/zero-pad to 400 frames
* fusion: each embedding is projected to 64-d, a softmax gate weights the two
  modalities (`w_voice + w_face = 1`), the weighted sum goes through a small
  MLP to a sigmoid score
* decision: `score >= 0.85` → ACCEPT (the only rule in the project, frozen
  from the original `config/settings.yaml`)

## Verified results (toy dataset, unchanged from the Phase 0 audit)

| population | pairs | mean | min | max | accept@0.85 |
| --- | --- | --- | --- | --- | --- |
| genuine (10 voices × 5 faces, enrolled identity) | 50 | 0.9182 | 0.8943 | 0.9293 | 1.00 |
| impostors, expanded 625-pair grid (Phase 1) | 625 | 0.0194 | 0.0078 | 0.0511 | 0.00 |

The two distributions do not overlap, which is expected for this toy setup:
there is exactly one enrolled identity and the "strangers" are synthetic
distractor features. A larger margin would say nothing about real-world
generalisation.

Reproduce it locally (needs the private data, which is not in the repository):

```
pytest tests/test_phase0_reference.py -v
python scripts/train/train_fusion.py --features-root outputs/features --epochs 1 --report-shipped
```

## Quickstart

```bash
# 1. install
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements/dev.txt      # runtime + pytest
#   or: pip install -r requirements/runtime.txt                  # runtime only

# 2. create synthetic example inputs (the repository ships no biometric data)
python examples/make_examples.py

# 3. verify one pair
python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav
python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav --json
python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav --backend pytorch

# 4. run the test suite
python -m pytest
#   -> 129 passed with the private enrolment data present
#   -> 125 passed, 4 skipped on a fresh clone (the reference tier needs data/)
```

The synthetic examples are not identity samples: they are a drawing and a
harmonic tone. Expect a low score and `REJECT` — that is the correct answer for
an input that is not the enrolled identity.

`onnxruntime>=1.24.1` is required — the INT8 graphs contain `ConvInteger`
nodes that older versions cannot even parse.

## Repository layout

```
configs/default.yaml            single source of truth for every parameter
src/multimodal_auth/            the importable package
  config/                       YAML loading + validation + cross-checks
  preprocessing/                face.py, audio.py (bit-exact to the audit)
  models/                       FaceExtractor, VoiceExtractor (verbatim copies)
  fusion/                       ModalAttentionFusion + attention introspection
  pipeline/                     onnx_backend, torch_backend, pipeline, types
  decision/                     threshold policy (ACCEPT/REJECT)
  safety.py                     write guards (protects weights/ and data/)
  cli.py, reporting.py          command line + human-readable report
scripts/                        runnable entry points (see the table below)
tests/                          pytest suite
examples/make_examples.py       synthetic, redistributable sample inputs
docs/                           audit trail, architecture, limits, manifest
weights/                        shipped artifacts (+ MODEL_ARTIFACT_MANIFEST)
requirements/                   runtime / dev / research (+ historical pins)
scripts_pc/                     FROZEN original scripts, kept as evidence
app_pi/                         HISTORICAL Raspberry Pi entry point
data/                           private biometric data (gitignored, never shipped)
```

## Commands

| command | purpose |
| --- | --- |
| `python scripts/infer.py --face F --voice V` | one verification; `--json`, `--threshold`, `--backend`, `--output`, `--fail-on-reject` |
| `python scripts/benchmark.py` | latency of every graph (INT8 vs fp32) + end-to-end; writes `outputs/benchmarks/<run>/` |
| `python scripts/data/01_build_voice_features.py` | raw audio → `outputs/features/voice/` (`--compare-to` for bit-equality) |
| `python scripts/data/02_build_face_features.py` | raw images → `outputs/features/face/` |
| `python scripts/train/train_extractors.py` | sandboxed re-training → `outputs/training/<run>/` |
| `python scripts/train/train_fusion.py` | sandboxed fusion re-training (`--report-shipped` scores the shipped head) |
| `python scripts/export.py` | PyTorch → ONNX fp32 → INT8 with numeric verification |
| `python -m pytest` | the full test suite |

Every script accepts `--help`. All of them write **only** under `outputs/`.

## Configuration

`configs/default.yaml` is the single source of truth: model contract, tensor
shapes, preprocessing parameters, runtime options and the decision threshold.
Values are frozen from the audit and validated on load, including cross-checks
(voice shape vs MFCC settings, face shape vs image size, embedding dims vs the
fusion input). Malformed or inconsistent configuration fails loudly with a
`ConfigError`; there is no silent fallback to a default.

Paths in the file are relative to the project root, which is discovered by
walking up from the config file, so the repository can be cloned and moved
anywhere.

## Safety rules (and why they exist)

During the original audit three `.pth` files were **overwritten**, because a
training script wrote back into its own input directory
([`docs/PHASE0_REMEDIATION_NOTICE.md`](docs/PHASE0_REMEDIATION_NOTICE.md)).
Phase 1 closes that whole class of bug:

* artifact-producing scripts may only write inside `<root>/outputs/`;
  `weights/`, `data/`, `src/`, `configs/`, `tests/` and `app_pi/` are refused
  (`src/multimodal_auth/safety.py`);
* they **fail closed**: an existing non-empty output directory is an error
  unless `--force` is given;
* [`docs/MODEL_ARTIFACT_MANIFEST.md`](docs/MODEL_ARTIFACT_MANIFEST.md) lists
  size + SHA-256 for all nine shipped files, and `tests/test_artifacts.py`
  reads that table — if any artifact changes, the suite fails;
* the legacy scripts in `scripts_pc/` are frozen and exit immediately unless
  `MULTIMODAL_AUTH_RUN_LEGACY=1` is set.

## Limitations

[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) is the authoritative list. In
short: one enrolled identity, synthetic strangers, no live capture, no
anti-spoofing or liveness detection, no bias evaluation, no Raspberry Pi
measurement, and INT8 is ~25× **slower** than fp32 on x86_64
(see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

## Provenance and audit trail

| document | content |
| --- | --- |
| [`docs/PHASE0_AUDIT_REPORT.md`](docs/PHASE0_AUDIT_REPORT.md) | what the original project actually did, and which claims hold |
| [`docs/PHASE0_REMEDIATION_NOTICE.md`](docs/PHASE0_REMEDIATION_NOTICE.md) | the checkpoint incident and exactly how it was repaired |
| [`docs/MODEL_ARTIFACT_MANIFEST.md`](docs/MODEL_ARTIFACT_MANIFEST.md) | size + SHA-256 + provenance of every shipped weight |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module map and data flow of the refactored code |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | how to reproduce every published number |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | measured latency on the development host |
| [`docs/TESTING.md`](docs/TESTING.md) | what the test suite covers and how to run it |
| [`docs/PRIVACY_AND_SECURITY.md`](docs/PRIVACY_AND_SECURITY.md) | biometric data handling, threat model, open-source risk |
| [`docs/HISTORICAL_EDGE_DEPLOYMENT.md`](docs/HISTORICAL_EDGE_DEPLOYMENT.md) | the Raspberry Pi story, and why it stays historical |
| [`docs/PHASE1_CHANGES.md`](docs/PHASE1_CHANGES.md) | exactly what Phase 1 changed, file by file |

## License

No license has been chosen for this repository yet. Before making it public,
pick one deliberately: the Haar cascade ships with OpenCV, the "stranger" face
images come from a research dataset, and some original data sources are
research-only. See [`docs/OPEN_SOURCE_RISK.md`](docs/OPEN_SOURCE_RISK.md).
Nothing here should be published under a permissive license by accident.


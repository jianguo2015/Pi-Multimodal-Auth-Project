# Multimodal Identity Authentication

**Lightweight face + voice identity verification with embedding-level attention fusion.**

Two modality-specific CNN encoders produce 128-dimensional embeddings. The
embeddings are combined through a learned attention-gated late fusion module.
The fused representation is passed to an MLP authentication head producing a
scalar score, which is compared against a single calibrated threshold.

```text
face image ─▶ FaceExtractor  ─▶ 128-d embedding ─┐
                                                 ├─▶ attention-gated fusion ─▶ score ─▶ ACCEPT/REJECT @ 0.85
voice clip ─▶ VoiceExtractor ─▶ 128-d embedding ─┘
```

> ### Status: research prototype — not a security product
>
> * The models were trained on a **toy dataset**: one enrolled identity (the
>   author) and five synthetic "strangers". Every number below describes that toy
>   setup and is **not** a production accuracy or FAR/FRR claim.
> * There is no liveness or anti-spoofing check, no live capture and no bias
>   evaluation. Do not wire this to a door, a lock or an account.
> * The evaluation is a prototype-level experiment, not a production biometric
>   benchmark. The reported genuine/impostor separation is based on the available
>   experimental dataset and should not be interpreted as production-grade
>   biometric accuracy or security.
>
> [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) is the authoritative list — read it
> before quoting anything from this repository.

## Contents

[What it is](#what-it-is) ·
[Architecture](#architecture) ·
[Quick start](#quick-start) ·
[Observed benchmark](#observed-benchmark) ·
[Experimental validation](#experimental-validation) ·
[Project status](#project-status) ·
[Repository layout](#repository-layout) ·
[Tests and validation contexts](#tests-and-validation-contexts) ·
[Configuration](#configuration) ·
[Safety rules](#safety-rules) ·
[Historical edge deployment](#historical-edge-deployment) ·
[Documentation](#documentation) ·
[License](#license)

## What it is

| | |
| --- | --- |
| **Task** | decide whether a face image and a voice clip come from the same enrolled identity |
| **Modalities** | face (Haar detection → largest face → 112×112 RGB crop) and voice (16 kHz mono → 40 MFCCs → 400 frames) |
| **Fusion** | learned attention gate over the two 128-d embeddings, then a small MLP head → scalar score |
| **Decision** | `score >= 0.85` → ACCEPT, else REJECT (the single rule shipped with the project) |
| **Parameters** | 423,236 (face) + 109,184 (voice) + 27,140 (fusion) ≈ 560 K parameters, 4.85 MB of weights |
| **Backends** | ONNX Runtime (default; the INT8 graphs) and PyTorch (research: exposes the attention weights) |
| **Runtime deps** | numpy, PyYAML, onnxruntime, opencv-python-headless, librosa, soundfile — torch only for the research paths |
| **License** | MIT for the code; the trained weights carry a separate research-use notice ([`MODEL_NOTICE.md`](MODEL_NOTICE.md)) |

No LLM, no RAG, no agent framework, no web UI — a small, self-contained
verification pipeline that fits in about 5 MB.

## Architecture

```text
                        face image
                            │
                            ▼
              Haar cascade detection ──▶ largest face ──▶ 112×112 crop
                            │
                            ▼
                    (x - 127.5) / 128 ──▶ ┌───────────────────────┐
                                          │ FaceExtractor (CNN)   │
                                          │ 423,236 parameters    │
                                          └───────────┬───────────┘
                                                      ▼
                                          128-d face embedding ──────────┐
                                                                         │
                        voice clip                                       │
                            │                                            │
                            ▼                                            │
        16 kHz mono ──▶ 40 MFCC ──▶ crop / zero-pad to 400 frames       │
                            │                                            │
                            ▼                                            │
                  ┌───────────────────────┐                              │
                  │ VoiceExtractor (CNN)  │                              │
                  │ 109,184 parameters    │                              │
                  └───────────┬───────────┘                              │
                              ▼                                          │
                    128-d voice embedding ───────────────────────────────┤
                                                                         ▼
                       ┌─────────────────────────────────────────────────────┐
                       │ ModalAttentionFusion (27,140 parameters)            │
                       │   project each 128-d embedding to 64-d,             │
                       │   softmax gate (w_voice + w_face = 1),              │
                       │   weighted sum ──▶ MLP ──▶ sigmoid                  │
                       └──────────────────────────┬──────────────────────────┘
                                                  ▼
                                            score ∈ [0, 1]
                                                  ▼
                              score >= 0.85 → ACCEPT, else REJECT
```

That diagram is the implementation rather than an aspiration: the module map is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and every number in it lives in
[`configs/default.yaml`](configs/default.yaml).

## Quick start

Requires **Python 3.11 or newer** — the pinned `onnxruntime` publishes no 3.10
wheel, so the earlier `>=3.10` claim was uninstallable
([`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md) §3.2).

```bash
git clone https://github.com/jianguo2015/multimodal-auth.git
cd multimodal-auth
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\python -m pip install -r requirements/dev.txt
# Linux / macOS: .venv/bin/python -m pip install -r requirements/dev.txt

# create the synthetic example inputs (the repository ships no biometric data)
python examples/make_examples.py

# score one (face, voice) pair
python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav
python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav --json
python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav --backend pytorch

# the public test tier and a benchmark run
python -m pytest -m "not private_data"
python scripts/benchmark.py --runs 30
```

Expected result on the synthetic examples: **`REJECT`, score ≈ 0.0466**. That is
the right answer for a drawing and a tone — they are not the enrolled identity —
and it says nothing about the model's accuracy.

Notes that save time:

* `requirements/dev.txt` = runtime + pytest; `requirements/runtime.txt` = runtime
  only; `requirements/research.txt` adds `torch`/`torchvision`/`onnx` for
  `--backend pytorch`, `scripts/export.py` and `scripts/train/*`.
* `opencv-python-headless` must stay `<5`: the 5.x wheels no longer ship
  `cv2/data/haarcascade_*.xml` (`docs/RELEASE_AUDIT.md` §3).
* `onnxruntime>=1.24.1` is required — the INT8 graphs contain `ConvInteger` nodes
  that older runtimes cannot parse at all.
* every script accepts `--help` and writes only under `outputs/`.

## Observed benchmark

Measured on one development laptop, not a general performance claim. Raw evidence
is committed: [`docs/benchmarks/20260922-development-host.{md,json}`](docs/benchmarks/20260922-development-host.json).

| item | value |
| --- | --- |
| host | Windows 11 (10.0.26200) x86_64, AMD Ryzen 9 7945HX, 16 cores / 32 threads, 15.7 GB RAM |
| GPU | NVIDIA GeForce RTX 4060 Laptop — **present but unused** (`CPUExecutionProvider` only) |
| Python / onnxruntime | 3.13.1 / 1.24.1, `intra_op_num_threads = 2` |
| inputs | `examples/sample_face.jpg` (512×512), `examples/sample_voice.wav` (2.0 s) |
| warm-up / runs | 5 / 30 per model (10 runs for the end-to-end row) |

| model | precision | file | size | mean | median | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| face encoder | INT8 | `face_extractor_quant.onnx` | 422.7 KB | 7.840 ms | 7.859 ms | 8.115 ms |
| voice encoder | INT8 | `voice_extractor_quant.onnx` | 114.5 KB | 25.218 ms | 25.441 ms | 26.448 ms |
| fusion head | INT8 | `attention_fusion_quant.onnx` | 39.2 KB | 0.021 ms | 0.021 ms | 0.024 ms |
| face encoder | fp32 | `face_extractor_fp32.onnx` | 1648.3 KB | 0.314 ms | 0.310 ms | 0.335 ms |
| voice encoder | fp32 | `voice_extractor_fp32.onnx` | 428.6 KB | 0.947 ms | 0.941 ms | 0.968 ms |
| fusion head | fp32 | `attention_fusion_fp32.onnx` | 109.5 KB | 0.016 ms | 0.016 ms | 0.017 ms |

| aggregate | value |
| --- | --- |
| three INT8 graphs | **33.321 ms** |
| three fp32 graphs | **1.267 ms** |
| INT8 / fp32 ratio | **26.30×** (a ratio above 1 means INT8 is slower) |
| end-to-end including preprocessing, INT8 | 59.973 ms median (60.520 ms mean) |
| peak RAM / model load time / cold start | **N/A — not measured** |

> On the tested x86 environment, the shipped INT8 model was substantially slower
> than FP32. This benchmark does not imply that INT8 is universally slower;
> runtime and hardware strongly affect quantized inference performance.

The original project claimed INT8 was "22× faster" — that claim was never
measured, and on this host the opposite holds. Quantisation here mainly reduces
size (1.9 MB → 0.6 MB for the three graphs). Preprocessing dominates the
end-to-end time; the three networks together need under 10 ms in fp32.

## Experimental validation

| population | pairs | mean | min | max | accept @ 0.85 |
| --- | --- | --- | --- | --- | --- |
| genuine (10 voices × 5 faces of the enrolled identity) | 50 | 0.918214 | 0.894259 | 0.929255 | 50 / 50 |
| impostors, expanded 625-pair grid | 625 | 0.0194 | 0.0078 | 0.0511 | **0 / 625** |

* **Boundary of these numbers.** The current evaluation is a prototype-level
  experiment, not a production biometric benchmark. There is exactly one enrolled
  identity and the "strangers" are five synthetic distractors, so the two
  distributions do not overlap. That is expected for this setup and says nothing
  about behaviour on real people.
* **Which impostor figure is published.** An earlier Phase 0 note quotes `0.0483`;
  that was the maximum over a 35-pair subset (one stranger modality at a time).
  The Phase 1 expansion to all 625 stranger/identity pairs raises the observed
  maximum to `0.0511`. Both are far below the 0.85 threshold, no impostor pair is
  accepted, and the conclusion is unchanged — but `0.0511` over 625 pairs is the
  figure this repository publishes.
* **INT8 vs PyTorch.** On the genuine grid the INT8 graphs give mean 0.9198 with
  50/50 accept; the largest per-pair score difference between the PyTorch and INT8
  paths is 0.0028. Do not mix backends when comparing results.
* These are **experimental validation** results, not accuracy: there is no
  train/validation/test split, no FAR/FRR/EER curve and no threshold calibration.

Reproduce (the first two need the private enrolment data, which is not published):

```bash
python scripts/train/train_fusion.py --features-root outputs/features --epochs 1 --report-shipped
pytest tests/test_phase0_reference.py -v      # private_data tier
python scripts/benchmark.py --runs 30         # latency, no private data needed
```

## Project status

```text
Available and verified
  ✅ PC inference              (ONNX Runtime INT8 default, PyTorch research path)
  ✅ Preprocessing reproducibility   (55/55 face tensors, 35/35 voice tensors bit-identical)
  ✅ Automated tests            (129 with data + research extras; 113 public, data-free)
  ✅ Artifact integrity checks  (size + SHA-256 for all nine shipped files)
  ✅ Benchmark                  (documented environment, warm-up, median, p95)
  ✅ Synthetic examples         (deterministic generator, no personal data)
  ✅ Configuration validation   (single YAML, cross-checked, relocatable)
  ✅ CI                         (public tier, end-to-end inference, artifact integrity)

Historical (provenance only, not re-validated)
  ⚠ Raspberry Pi 4B deployment path   (app_pi/, dummy_capture, no Pi measurement exists)

Not currently provided
  ✗ live camera/microphone capture
  ✗ liveness / anti-spoofing
  ✗ multi-identity enrolment or FAR/FRR evaluation
  ✗ Raspberry Pi re-validation
  ✗ production biometric security evaluation
```

## Repository layout

```text
.github/workflows/ci.yml      public validation workflow (Ubuntu, python 3.11 + 3.13)
src/multimodal_auth/          the package: importable, installable, torch-free by default
├── config/loader.py          YAML -> frozen dataclasses, validated and cross-checked on load
├── preprocessing/            face.py (Haar -> 112x112) and audio.py (MFCC -> 400 frames)
├── models/                   FaceExtractor, VoiceExtractor (byte-identical to the audited originals)
├── fusion/                   ModalAttentionFusion + attention introspection
├── decision/policy.py        score + threshold -> ACCEPT / REJECT
├── pipeline/                 onnx_backend, torch_backend, pipeline, JSON-safe result types
├── safety.py                 write sandbox, path hygiene, hashing
├── reporting.py              the human-readable report
└── cli.py                    argparse front end and exit codes

scripts/                      supported entry points (all write only under outputs/)
├── infer.py                  one (face, voice) pair -> score + decision
├── benchmark.py              latency measurement of the shipped artifacts
├── export.py                 .pth -> fp32 ONNX -> dynamic INT8
├── data/                     build features from raw media, non-destructively
└── train/                    re-train the two extractors and the fusion head

scripts_pc/                   FROZEN Phase 0 pipeline: provenance only, exits unless authorised
app_pi/                       HISTORICAL Raspberry Pi adapter, explicitly not supported
configs/default.yaml          the single validated configuration
config/settings.yaml          the original Phase 0 configuration (provenance)
weights/onnx/                 3 fp32 + 3 INT8 graphs (the INT8 trio is the default)
weights/pytorch/              3 state_dicts (research, export, attention inspection)
examples/make_examples.py     deterministic synthetic inputs (seed 20260419)
tests/                        10 modules, 129 tests, public + private_data tiers
docs/                         audit trail, architecture, benchmarks, limitations, release records
requirements/                 runtime.txt, dev.txt, research.txt, historical/
LICENSE  MODEL_NOTICE.md  THIRD_PARTY_NOTICES.md  CHANGELOG.md  pyproject.toml
```

`docs/CODEBASE_MAP.md` describes the *pre-Phase-1* layout on purpose: it is the
Phase 0 provenance record, kept verbatim except for the redaction of local paths.

## Tests and validation contexts

```text
129 tests total = 125 public (no biometric data needed) + 4 reference tests (private_data)
```

| context | command | measured result |
| --- | --- | --- |
| **Local full validation** (runtime + test + research extras, enrolment data present) | `python -m pytest` | **129 passed** |
| **Public clone, runtime + test requirements only** (no torch) | `python -m pytest -m "not private_data"` | **113 passed, 5 skipped, 4 deselected** |
| **Public clone**, same environment, whole suite | `python -m pytest` | **113 passed, 9 skipped** |
| **Public clone with research extras** (torch, no enrolment data) | `python -m pytest` | **125 passed, 4 skipped** |
| **Private data validation** (the `private_data` tier, needs `data/`) | `python -m pytest tests/test_phase0_reference.py -v` | 4 passed locally, reproducing the published grids |
| **CI** (Ubuntu, both public combinations) | `.github/workflows/ci.yml` | same numbers as the two public rows |

`docs/FRESH_CLONE_VALIDATION.md` names **every** skipped test and its reason. The
skips are not silent: a module that cannot import torch is reported as one skip
instead of a collection error, and CI asserts that the four reference tests are
skipped for the missing biometric data and for nothing else.

What the public tier covers: config validation, preprocessing (shapes, ranges,
determinism, failure modes), model artifact integrity (size + SHA-256 of all nine
files), the decision rule, fusion numerics, the CLI contract (exit codes, JSON),
the safety sandbox, and end-to-end inference on the synthetic examples.

## Configuration

`configs/default.yaml` is the single source of truth: model contract, tensor
shapes, preprocessing parameters, runtime options and the decision threshold.
Every value is validated on load, including cross-checks (voice shape vs MFCC
settings, face shape vs image size, embedding dims vs the fusion input).
Malformed or inconsistent configuration fails loudly with a `ConfigError`; there
is no silent fallback to a default.

Paths are relative to the project root, which is found by walking up from the
config file — so the repository can be cloned or moved anywhere.

```bash
python scripts/infer.py --show-config          # the effective configuration
python scripts/infer.py --face ... --voice ... --threshold 0.90   # explicit override
```

## Safety rules

During the Phase 0 audit three `.pth` files were **overwritten**, because a
training script wrote back into its own input directory
([`docs/PHASE0_REMEDIATION_NOTICE.md`](docs/PHASE0_REMEDIATION_NOTICE.md)).
This release closes that whole class of bug:

* artifact-producing scripts may only write inside `<root>/outputs/`; `weights/`,
  `data/`, `src/`, `configs/`, `tests/` and `app_pi/` are refused
  (`src/multimodal_auth/safety.py`);
* they **fail closed**: an existing non-empty output directory is an error unless
  `--force` is given;
* [`docs/MODEL_ARTIFACT_MANIFEST.md`](docs/MODEL_ARTIFACT_MANIFEST.md) pins size
  and SHA-256 of all nine shipped files, and `tests/test_artifacts.py` reads that
  table — if a single byte changes, the suite fails;
* the legacy pipeline in `scripts_pc/` is frozen and exits immediately unless
  `MULTIMODAL_AUTH_RUN_LEGACY=1` is set;
* reports never embed machine-specific absolute paths, so a copied JSON cannot
  leak a local user name.

## Historical edge deployment

`app_pi/` is a **historical Raspberry Pi 4B deployment adapter**, not a supported
entry point. Its capture path is a `np.random.randn` placeholder (`dummy_capture`)
and no Raspberry Pi measurement exists anywhere in this repository — so no Pi
latency, FPS, CPU, RAM or temperature number is claimed.

Two real defects from that era are documented rather than hidden: the historical
requirement pin `onnxruntime==1.16.0` cannot even load the shipped INT8 graphs,
and the "INT8 is 22× faster" claim was never measured (the opposite holds on the
tested x86 host). Details:
[`docs/HISTORICAL_EDGE_DEPLOYMENT.md`](docs/HISTORICAL_EDGE_DEPLOYMENT.md).

The supported way to run this project is PC inference:
`python scripts/infer.py --face ... --voice ...`.

## Documentation

| document | content |
| --- | --- |
| [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md) | the open-source audit: asset inventory, licenses, datasets, secret scan, release gates |
| [`docs/MODEL_RELEASE.md`](docs/MODEL_RELEASE.md) | what the weights are, where they came from, and the terms they ship under |
| [`docs/FRESH_CLONE_VALIDATION.md`](docs/FRESH_CLONE_VALIDATION.md) | clone validation and every skipped test, named |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | the authoritative limitations list |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module map, data flow, backends, extension points |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | how to reproduce every published number, and what cannot be reproduced |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | the latency measurement and its environment |
| [`docs/TESTING.md`](docs/TESTING.md) | what the suite covers, per environment |
| [`docs/PRIVACY_AND_SECURITY.md`](docs/PRIVACY_AND_SECURITY.md) | biometric data handling, threat model, what is not defended |
| [`docs/HISTORICAL_EDGE_DEPLOYMENT.md`](docs/HISTORICAL_EDGE_DEPLOYMENT.md) | the Raspberry Pi story, and why it stays historical |
| [`docs/MODEL_ARTIFACT_MANIFEST.md`](docs/MODEL_ARTIFACT_MANIFEST.md) | size + SHA-256 + provenance of every shipped weight |
| [`docs/PHASE0_AUDIT_REPORT.md`](docs/PHASE0_AUDIT_REPORT.md), [`docs/PHASE0_REMEDIATION_NOTICE.md`](docs/PHASE0_REMEDIATION_NOTICE.md) | what the original project actually did, and the checkpoint incident |
| [`docs/PHASE1_CHANGES.md`](docs/PHASE1_CHANGES.md) | the Phase 1 engineering pass, file by file |
| [`docs/OPEN_SOURCE_RISK.md`](docs/OPEN_SOURCE_RISK.md), [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md), [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md), [`docs/REPRODUCTION_STATUS.md`](docs/REPRODUCTION_STATUS.md) | Phase 0 provenance records (local paths redacted) |

## License

* **Code, configuration, scripts, tests and documentation: MIT** — see
  [`LICENSE`](LICENSE). Chosen because every runtime dependency is permissively
  licensed and because MIT grants no patent rights, which leaves the author's
  position on the fusion method untouched (rationale and rejected alternatives in
  [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md) §8).
* **Trained weights (`weights/`): research / evaluation use, non-commercial** — see
  [`MODEL_NOTICE.md`](MODEL_NOTICE.md). They are not covered by MIT, because the
  networks were trained with the help of a research-use face database.
* **Third-party components and datasets** (OpenCV's Haar cascade, the
  Olivetti/ORL database, the Python dependencies): see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Nothing in this repository is certified, and no part of it should be presented as
production-ready biometric security.





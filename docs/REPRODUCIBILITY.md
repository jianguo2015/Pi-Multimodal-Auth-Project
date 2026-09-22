# Reproducibility

Everything in this repository is meant to be re-derivable. This page states
what is reproducible, how, and what is not.

## 1. Environment

| item | value used for the published numbers |
| --- | --- |
| OS | Windows 11 (10.0.26200) x86_64 |
| Python | 3.13.1 |
| onnxruntime | 1.24.1 (CPUExecutionProvider) |
| numpy / opencv / librosa / soundfile | 2.x / 4.13 / 0.11 / 0.13 (any recent version works) |
| torch (research paths only) | 2.7.1+cu118 |

Install: `pip install -r requirements/dev.txt` (add
`-r requirements/research.txt` for torch/onnx).

## 2. Reproduce the accuracy numbers

Requires the private enrolment data (`data/processed/`), which is **not**
shipped — it contains the author's biometric data.

```bash
python scripts/data/01_build_voice_features.py --compare-to data/processed/voice_features
python scripts/data/02_build_face_features.py  --compare-to data/processed/face_features
python -m pytest tests/test_phase0_reference.py -v
```

Expected (and asserted by the tests):

| measurement | value |
| --- | --- |
| voice features bit-identical to the stored set | 35 / 35 |
| face features bit-identical to the stored set | 55 / 55 |
| genuine grid (10 voices × 5 faces) | mean 0.9182, min 0.8943, max 0.9293 |
| genuine accept rate @0.85 | 50 / 50 |
| impostor grid (625 pairs, expanded in Phase 1) | max 0.0511 |
| impostor accept rate @0.85 | 0 / 625 |

The single-pair check is also a script:

```bash
python scripts/train/train_fusion.py --features-root outputs/features --epochs 1 --report-shipped
# -> shipped fusion head on the genuine grid: {'pairs': 50, 'mean': 0.918214, ...}
```

## 3. Reproduce the INT8 / fp32 relationship

```bash
python scripts/benchmark.py --runs 30          # ~1 minute
```

Published table: `docs/BENCHMARKS.md`; committed raw evidence:
`docs/benchmarks/20260922-development-host.{md,json}`. Expect the INT8 graphs to
be roughly **25× slower** than fp32 on x86_64 — the opposite of the original
report's unsupported claim.

## 4. Reproduce the artifacts

`scripts/export.py` rebuilds fp32 ONNX and dynamic INT8 from the `.pth`
checkpoints and reports the numeric deltas (`fp32 vs pytorch`,
`int8 vs fp32`). It never overwrites `weights/`.

Rebuilding does **not** reproduce the shipped bytes: the original export ran
with a different PyTorch version, and the reported numbers differ in the last
digits. Functional equivalence is what is verified, not byte equality. The
shipped files themselves are protected by
`docs/MODEL_ARTIFACT_MANIFEST.md` + `tests/test_artifacts.py`.

## 5. What cannot be reproduced

| thing | why |
| --- | --- |
| the exact training runs | the original scripts seeded nothing and consumed data that is not published; `scripts/train/*` reproduces the *recipe* only |
| the original `.pth` bytes | they were destroyed by the Phase 0 accident and rebuilt from the fp32 ONNX graphs; see `docs/PHASE0_REMEDIATION_NOTICE.md` |
| the face extractor's BatchNorm statistics | folded into the preceding `Conv` by the ONNX exporter, therefore not recoverable (the composed function is reproduced to 5.36e-07) |
| Raspberry Pi latency | never measured, no device |
| real-world accuracy / FAR / FRR | only one enrolled identity and synthetic strangers exist |
| the "22× faster INT8" claim | it is not true and was never measured |

## 6. Reproducibility hygiene in the code

* preprocessing is deterministic (no random state, no timestamps);
* the pipeline is pure with respect to its inputs: the same files give bit-identical
  scores on repeated runs (`tests/test_pipeline.py::test_inference_is_deterministic`);
* features are stored as raw `float32` `.npy` so bit equality is checkable with
  `np.array_equal`, not `allclose`;
* training scripts take `--seed` and record it in `run.json`;
* benchmark and training outputs land under `outputs/<kind>/<run-name>/` and
  refuse to overwrite an existing non-empty run directory.

## 7. Verification log of the Phase 1 refactor

| check | result |
| --- | --- |
| model definitions byte-identical to the audited originals | 3 / 3 (SHA-256) |
| face preprocessing vs the original `utils.extract_face` (ASCII path) | 55 / 55 bit-identical |
| face preprocessing vs the stored Phase 0 features (real path) | 55 / 55 bit-identical |
| voice preprocessing vs `utils.extract_mfcc` and the stored features | 35 / 35 bit-identical |
| genuine grid, PyTorch | mean 0.918214 / min 0.894259 / max 0.929255 |
| genuine grid, INT8 ONNX | mean 0.9198, accept 50/50 |
| impostor grid, INT8 ONNX (625 pairs) | max 0.0511, accept 0/625 |
| `weights/` hashes before vs after running every script | unchanged |
| test suite | see `docs/TESTING.md` |

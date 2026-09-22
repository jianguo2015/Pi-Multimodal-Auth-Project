# Testing

## Running the suite

```bash
python -m pip install -r requirements/dev.txt   # runtime + pytest
python -m pytest                                # everything
python -m pytest -v                             # per-test names
python -m pytest tests/test_phase0_reference.py -v   # only the reference numbers
python -m pytest -m "not private_data"          # only the public, data-free part
```

`pyproject.toml` sets `testpaths = ["tests"]` and `--strict-markers`.

## Two tiers

| tier | needs | behaviour without the data |
| --- | --- | --- |
| public | nothing but the shipped artifacts and a writable temp dir | always runs |
| `private_data` | `data/processed/.../User_Me`+ strangers | automatically **skipped** (see `pytest_collection_modifyitems` in `tests/conftest.py`) |

A fresh clone therefore gets a green suite, while a machine holding the
enrolment data additionally verifies the published accuracy numbers.

The suite never writes outside `tmp_path` or `<root>/outputs/tests-sandbox`
(cleaned up), and it never reads `data/` unless the private tier is active.

## What is covered

| file | tests | focus |
| --- | --- | --- |
| `test_config.py` | 21 | frozen values, tensor contracts, root discovery, relocatability, JSON export, and the failure modes (missing key, bad type, out-of-range threshold, unknown backend, inconsistent shapes, zero scale, non-positive threads) |
| `test_preprocessing.py` | 18 | face shape/dtype/range/determinism, array vs path input, centre-crop fallback, missing/empty/undecodable/wrong-channel images, non-ASCII round trip; audio shape, zero-padding, cropping, resampling, and missing/empty/undecodable/too-short clips |
| `test_safety.py` | 18 | `outputs/`-only writes, protected directories refused, `..` and absolute subdirs refused, fail-closed on non-empty output dirs, hashing helpers |
| `test_artifacts.py` | 19 | every file in `weights/` matches `MODEL_ARTIFACT_MANIFEST.md` (size + SHA-256), no stray files, ONNX loads under the pinned onnxruntime, input names/order and dynamic batch axis, `.pth` files load strictly |
| `test_pipeline.py` | 15 | lazy loading, `describe()`, full result contract, JSON serialisation without raw vectors, determinism, threshold override, missing/unknown artifacts, ONNX↔PyTorch agreement (score < 1e-2, embedding cosine > 0.99) |
| `test_decision.py` | 16 | inclusive boundary (`score == threshold` accepts), margin sign, the Phase 0 operating points, rejection of out-of-range scores/thresholds |
| `test_cli.py` | 10 | exit codes (0/2/3/4/5), human report contents, `--json`, `--show-config`, `--fail-on-reject`, `--output`, refusal to write results into `weights/` |
| `test_fusion.py` | 8 | the shipped checkpoint loads `strict=True`; `w_voice + w_face == 1`; the introspection helper reproduces `forward` to 1e-7; the fused feature is the weighted sum; determinism in eval mode; training flag restored |
| `test_phase0_reference.py` | 4 | `private_data`: genuine grid = 0.9182 / 0.8943 / 0.9293 with 100 % accept, 625-pair impostor grid with 0 % accept, raw→feature bit-exactness, ONNX↔torch agreement on real features |
| **total** | **129** | |

## The reference tier in detail

`test_phase0_reference.py::test_genuine_grid_matches_phase0` is the strongest
assertion in the repository: it rebuilds the 10×5 genuine score grid from the
stored Phase 0 features and compares the mean/min/max to the published values
within 1e-3:

```
published   mean 0.9182  min 0.8943  max 0.9293
asserted    mean ±1e-3   min ±1e-3   max ±1e-3
```

If a preprocessing change, a weight change or a backend change breaks the
published numbers, this test fails first.

`test_preprocessing_is_bit_exact_on_raw_inputs` re-derives features from the raw
media and compares with `np.testing.assert_array_equal` — bit equality, not
`allclose`.

## Testing philosophy

* **Fail closed, never guess.** Tests assert that invalid input *raises*, rather
  than asserting some fallback value.
* **Boundaries are explicit.** The threshold rule is tested exactly at 0.85.
* **Confidence intervals over point claims.** Latency is never asserted in
  tests; it is measured by `scripts/benchmark.py` and published in
  `docs/BENCHMARKS.md`.
* **No network, no clock, no randomness** in the public tier: sample inputs are
  generated deterministically from a seed.

## Not covered

* Live capture (camera/microphone) — the code does not implement it.
* Raspberry Pi behaviour — no device available.
* `app_pi/` and `scripts_pc/` — historical code, deliberately frozen.
* Any real-world accuracy claim — the data does not exist for it.
* There is **no CI configuration** in this repository yet. Before publishing,
  add a workflow that runs `python -m pytest -m "not private_data"` on
  Linux/Windows with `requirements/dev.txt`; the private tier cannot run there
  by design.

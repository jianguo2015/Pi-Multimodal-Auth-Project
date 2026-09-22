# Fresh-clone validation

Proves that a public clone runs **without** the biometric data, in the
environments a stranger actually has. Phase 1 summarised the clone as
"125 passed, 4 skipped"; Phase 2 checked that claim test by test and found it
described only one of the environments (§4). Every number below is measured, not
derived, and the per-test skip tables are complete.

## 1. What was cloned, and into what

```bash
git clone <release repository> C:/AI/multimodal_auth_fresh    # ASCII-only path, on purpose
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements/dev.txt   # the documented install
```

* the clone contains no `data/` and no `outputs/` directory (verified), so the
  private enrolment tier cannot run there — which is the point;
* the clone contains no `.idea/`, no `my_gan_data/`, no coursework files and no
  Phase 0 audit cache, i.e. none of the gitignored working-tree material;
* two environments were measured against that same clone:

| environment | installed | torch |
| --- | --- | --- |
| **plain** — "install what the README says" | `requirements/dev.txt` only | no |
| **research** — the same plus the research extras | `requirements/dev.txt` + `torch` (CPU wheel, 2.14.0+cpu) | yes |

Counts were read from `pytest --junitxml` rather than from the console, because
pytest 9 in this configuration does not print its final summary line when output
is redirected or captured (an annoyance, not a failure).

## 2. Measured results

| environment | command | collected | passed | skipped | failed | errors | deselected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plain | `python -m pytest` | 122 | **113** | 9 | 0 | 0 | 0 |
| plain | `python -m pytest -m "not private_data"` | 118 | **113** | 5 | 0 | 0 | 4 |
| research | `python -m pytest` | 129 | **125** | 4 | 0 | 0 | 0 |
| research | `python -m pytest -m "not private_data"` | 125 | **125** | 0 | 0 | 0 | 4 |

Exit code 0 in all four cases. The two `collected` totals differ (122 vs 129)
because in the plain environment `tests/test_fusion.py` is a single module-level
skip instead of eight collected tests — the same reason the public command
reports 113 passed there rather than 121.

For comparison, the same suite on the author's machine, where the research extras
**and** the enrolment data are present: **129 passed, 0 skipped**
(`docs/TESTING.md`).

## 3. Every skipped test, named

### 3.1 Without the research extras (plain environment, both commands)

| test | reason reported by pytest |
| --- | --- |
| `tests/test_fusion.py` (module, 8 tests) | `collection skipped` — `pytest.importorskip("torch")` at module level |
| `tests/test_artifacts.py::test_pytorch_checkpoints_load_strictly` | `could not import 'torch'` |
| `tests/test_cli.py::test_pytorch_backend_selection` | `could not import 'torch'` |
| `tests/test_pipeline.py::test_torch_backend_agrees_with_onnx` | `could not import 'torch'` |
| `tests/test_pipeline.py::test_torch_backend_embeddings_track_onnx` | `could not import 'torch'` |
| `tests/test_phase0_reference.py` (4 tests, deselected by `-m "not private_data"`, skipped otherwise) | `the reference tier needs requirements/research.txt (torch)` |

### 3.2 With the research extras (research environment)

| test | reason reported by pytest |
| --- | --- |
| `tests/test_phase0_reference.py::test_genuine_grid_matches_phase0` | `private biometric data not present on this machine` |
| `tests/test_phase0_reference.py::test_impostors_are_rejected` | `private biometric data not present on this machine` |
| `tests/test_phase0_reference.py::test_preprocessing_is_bit_exact_on_raw_inputs` | `private biometric data not present on this machine` |
| `tests/test_phase0_reference.py::test_onnx_pipeline_agrees_with_pytorch_on_real_features` | `private biometric data not present on this machine` |

**These four are the only skips a user of the published repository will ever see
in a fully installed environment**, and they are skipped for exactly one reason:
the biometric enrolment data is not published. CI asserts that reason explicitly
(`.github/workflows/ci.yml`, job `research-tier`), so it cannot silently change
into "skipped because the import broke".

## 4. What this validation exposed (all fixed in this release)

| finding | evidence | fix |
| --- | --- | --- |
| `pytest -m "not private_data"` — the documented public command — **failed during collection** in a clean environment: `ModuleNotFoundError: No module named 'torch'`, `Interrupted: 1 error during collection` | measuring the plain environment before the fix | `tests/test_phase0_reference.py` defers its torch import and is skipped instead of erroring (commit `342e7ea`) |
| the feature-building scripts referenced by `docs/REPRODUCIBILITY.md` (`scripts/data/*.py`) were **not in the repository** | `git status --porcelain --ignored` → `!! scripts/data/`; the unanchored `.gitignore` pattern `data/` matched them | `.gitignore` patterns anchored to the repository root; the three scripts are now tracked (`342e7ea`) |
| "125 passed, 4 skipped" described one environment and silently implied the others | the four measured combinations in §2 | `docs/TESTING.md`, `docs/PHASE1_CHANGES.md` and the README now state all of them |
| the declared Python floor `>=3.10` was uninstallable | the pinned `onnxruntime==1.24.1` ships cp311–cp314 wheels only | `requires-python = ">=3.11"`, enforced by the CI matrix |
| an unpinned `opencv-python-headless` broke every face input | 5.x wheels ship no `cv2/data/` and no `cv2.CascadeClassifier` | pinned `<5` in all requirement files |

## 5. End-to-end checks in the clone

| step | command | observed |
| --- | --- | --- |
| example generator (idempotent) | `python examples/make_examples.py` | `exists, keeping (use --force to regenerate)` ×2, exit 0 |
| one verification | `python scripts/infer.py --face examples/sample_face.jpg --voice examples/sample_voice.wav` | `REJECT`, score **0.046629**, margin −0.803371, exit 0 |
| JSON report | the same with `--json` | parses; `model_files` are **repository-relative** (`weights/onnx/face_extractor_quant.onnx`), no absolute path anywhere |
| human report | as above | `Face input : examples/sample_face.jpg`, `Config : configs/default.yaml` — no absolute path |
| benchmark smoke | `python scripts/benchmark.py --warmup 1 --runs 5` | 34.145 ms INT8 vs 1.405 ms fp32 → ratio 24.31×; end-to-end median 63.9 ms |
| artifact integrity | `pytest tests/test_artifacts.py` | all nine files match `docs/MODEL_ARTIFACT_MANIFEST.md` (included in the 125 passed) |

Two things worth pointing out:

* the fusion score **0.046629 is identical to the score produced by the original
  development checkout** for the same inputs. The repository is genuinely
  relocatable: different drive, different directory depth, different path
  characters, same number.
* the benchmark smoke figure (24.31×) is a 5-run smoke test on a different day,
  not a replacement for the committed 30-run evidence (26.30×) in
  `docs/benchmarks/`. Small variations of this order are expected; the conclusion
  (INT8 slower than fp32 on x86_64) is stable across both.

Observed but harmless: with the much newer `torch 2.14.0+cpu`,
`tests/test_fusion.py::test_attention_breakdown_matches_forward` prints a
`UserWarning` about converting a `requires_grad=True` tensor to a scalar. It is a
warning, not a failure, and the test passes.

## 6. What this validation does *not* cover

* **The GitHub-hosted clone.** The repository had not been pushed yet when this
  was run, so the clone was taken from the local release repository at the same
  commit. The post-push check is the only step that cannot be done locally.
* **Linux / GitHub Actions.** `.github/workflows/ci.yml` runs the same commands,
  but the workflow itself can only be exercised by GitHub after the first push;
  here it was verified on Windows only.
* **The private tier's assertions.** Those need `data/`, which is not published —
  they pass on the author's machine (`129 passed`), and CI only asserts that they
  are skipped for the documented reason.
* **Raspberry Pi.** No device, no measurement (`docs/HISTORICAL_EDGE_DEPLOYMENT.md`).


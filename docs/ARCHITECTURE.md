# Architecture

## Design constraints inherited from Phase 0

1. **No model change.** The three networks, their weights and the preprocessing
   maths must stay bit-for-bit identical. Any refactor had to prove equality
   numerically, not argue it.
2. **One source of truth for configuration.** The original project read the same
   YAML from five scripts with `../config/settings.yaml` relative paths, and
   hard-coded the numbers again inside `utils/` and `models/`.
3. **Fail loudly.** The original returned `None` on unreadable input
   (`extract_face`, `extract_mfcc`), which silently produced empty predictions.
4. **Never write into the inputs.** The Phase 0 incident
   (`docs/PHASE0_REMEDIATION_NOTICE.md`) must not be repeatable.

## Module map

```
multimodal_auth
├── errors.py            exception hierarchy (ConfigError, InvalidImageError, ...)
├── safety.py            write guards + hashing (no ML imports at all)
├── config/
│   ├── loader.py        YAML -> frozen dataclasses, full validation
│   └── __init__.py      re-exports
├── preprocessing/
│   ├── face.py          image -> (3,112,112) float32
│   └── audio.py         audio -> (1,40,400) float32
├── models/              FaceExtractor, VoiceExtractor  (byte-identical copies)
├── fusion/
│   ├── attention_fusion.py   ModalAttentionFusion      (byte-identical copy)
│   └── inspect_attention.py  score + gate weights, without touching forward()
├── decision/policy.py   score + threshold -> ACCEPT / REJECT
├── pipeline/
│   ├── onnx_backend.py  ONNX Runtime sessions (default)
│   ├── torch_backend.py PyTorch modules (research, exposes attention)
│   ├── pipeline.py      preprocess -> embed -> fuse -> decide
│   └── types.py         AuthResult / EmbeddingSummary (JSON-safe)
├── reporting.py         human-readable report text
└── cli.py               argparse front end + exit codes
```

### Dependency direction

```
cli ─▶ pipeline ─▶ {onnx_backend, torch_backend} ─▶ models, fusion
             └──▶ preprocessing ─▶ config
decision, errors, safety      ← no upward dependencies
```

`multimodal_auth` itself imports **no** torch. `models/__init__.py` and
`fusion/__init__.py` resolve `FaceExtractor`, `VoiceExtractor` and
`ModalAttentionFusion` lazily through PEP 562 `__getattr__`, so the ONNX-only
install stays light and `import multimodal_auth` cannot drag in a 2 GB wheel
chain.

## Data flow of one verification

| step | code | input | output |
| --- | --- | --- | --- |
| 1 | `preprocess_face` | image path or BGR array | `(3,112,112)` float32, roughly `[-1,1]` |
| 2 | `preprocess_voice` | audio path or waveform | `(1,40,400)` float32 |
| 3 | `engine.embed_face` | tensor | `(128,)` float32 |
| 4 | `engine.embed_voice` | tensor | `(128,)` float32 |
| 5 | `engine.score` | the two embeddings | scalar in `[0,1]` (+ gate weights on the torch path) |
| 6 | `decision.evaluate` | score | `DecisionResult(ACCEPT/REJECT, margin)` |
| 7 | `AuthResult.as_dict` | everything | JSON that never contains raw vectors |

Steps 1–4 are identical for both backends, which is what makes
`tests/test_pipeline.py::test_torch_backend_agrees_with_onnx` meaningful.

## Why the copies are literal

`models/face_extractor.py`, `models/voice_extractor.py` and
`fusion/attention_fusion.py` are byte-identical copies of the audited originals
(verified by SHA-256 during the refactor). Two consequences:

* `load_state_dict(..., strict=True)` keeps working, so the shipped `.pth`
  files remain loadable and `torch.onnx.export` remains reproducible;
* the numerical pipeline cannot drift by accident — there is no "cleaned up"
  version of the model code.

The only additions are *around* the models: `fusion/inspect_attention.py`
replays the same submodules in the same order to expose the attention gate,
and `tests/test_fusion.py` asserts that the replayed score equals
`ModalAttentionFusion.forward` to 1e-7.

## The two backends

| | ONNX (`--backend onnx`, default) | PyTorch (`--backend pytorch`) |
| --- | --- | --- |
| artifacts | `weights/onnx/*_quant.onnx` (INT8) | `weights/pytorch/*.pth` |
| extra dependency | `onnxruntime` | `torch` |
| attention weights | not available | available |
| speed on x86_64 | ~33 ms for the 3 graphs | ~1.3 ms |
| used by | `scripts/infer.py`, `scripts/benchmark.py` | introspection, `scripts/export.py` |

## Configuration model

`load_config()` returns a frozen `Config` dataclass tree. Everything is
validated at load time:

* required keys, types and ranges;
* cross-field consistency: `voice.input_shape == (1, n_mfcc, max_pad_len)`,
  `face.input_shape == (3, image_size, image_size)`,
  `face.embed_dim == voice.embed_dim`,
  `fusion.input_shape == (face.embed_dim, voice.embed_dim)`;
* `runtime.backend ∈ {onnx, pytorch}`;
* `0 < threshold < 1`.

The project root is found by walking up from the config file until a directory
containing both `configs` and `weights` is seen, so relative paths resolve the
same way regardless of the current working directory. That is what makes
`tests/test_config.py::test_relocatable_copy` possible.

## Extension points

* **new preprocessing step** → add a function in `preprocessing/`, take the
  sub-config as an argument, return a C-contiguous `float32` array, raise
  `InvalidInputError` subclasses on bad input, add a test.
* **new backend** → implement `embed_face`, `embed_voice`, `score`,
  `model_files` and register it in `pipeline.create_backend`.
* **new decision rule** → extend `decision/policy.py`; keep the default
  `score >= threshold` untouched so published numbers stay comparable.
* **new script** → write only through `safety.prepare_output_dir`, add a
  `--help` and a test that it cannot write into `weights/`.

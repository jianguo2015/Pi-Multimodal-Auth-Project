# Model Artifact Manifest

Every file shipped under `weights/` is listed here with its exact size and
SHA-256. The table below is **machine-read by `tests/test_artifacts.py`**; if a
single byte of any artifact changes, the test suite fails. That is deliberate:
during Phase 0 three `.pth` files were overwritten by a training script that
wrote back into its own input directory
(see `docs/PHASE0_REMEDIATION_NOTICE.md`). This manifest plus the sandbox in
`src/multimodal_auth/safety.py` are the fix.

Verify at any time:

```
python -c "import hashlib,pathlib; p=pathlib.Path('weights/onnx/face_extractor_quant.onnx'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
pytest tests/test_artifacts.py -v
```

## 1. Inventory

| file | bytes | sha256 | format | role | provenance |
| --- | --- | --- | --- | --- | --- |
| `weights/onnx/attention_fusion_fp32.onnx` | 112106 | `d8272a3986bb773502d395ff628867c1100805397b0b75ba9619458f3233975c` | ONNX, opset 11, fp32 | reference graph (not used by the default config) | original, untouched |
| `weights/onnx/attention_fusion_quant.onnx` | 40172 | `34c2c0586a69018d95b2a95d110bffce95e94cf911dcb15e10f33899b77d5bc0` | ONNX, dynamic INT8 | canonical fusion head | original, untouched |
| `weights/onnx/face_extractor_fp32.onnx` | 1687815 | `3949be0816c813a9da2b3cba82d1d601f62e4e12501d0d3eb5a1711126c76345` | ONNX, opset 11, fp32 | reference graph | original, untouched |
| `weights/onnx/face_extractor_quant.onnx` | 432865 | `0ba342ddf0b5ee345588985fbffcb4ea3a4164972d6df0c3bf99aaa7738e51a2` | ONNX, dynamic INT8 | canonical face extractor | original, untouched |
| `weights/onnx/voice_extractor_fp32.onnx` | 438936 | `191697363be7eb0ad51b669922e8dd0caa6a6dccd500bc4437f8f7a575a25c8a` | ONNX, opset 11, fp32 | reference graph | original, untouched |
| `weights/onnx/voice_extractor_quant.onnx` | 117288 | `ec25946393150e25eadf6e8c263ec9215cc2e94c600d18273f76f2ec2425083b` | ONNX, dynamic INT8 | canonical voice extractor | original, untouched |
| `weights/pytorch/attention_fusion_weights.pth` | 114019 | `a3145830354b5db772b30733ad7176338930db9b29cdc0a78fd11f3da2e40540` | PyTorch state_dict | fusion head (pytorch backend) | REBUILT by Phase 0, parameter-identical |
| `weights/pytorch/face_extractor_weights.pth` | 1701820 | `574fb82d9835f2bbc8f80617e92ac2e81814667fc360b2df9864b2c7dfcba6a7` | PyTorch state_dict | face extractor (pytorch backend) | REBUILT by Phase 0, functionally equivalent |
| `weights/pytorch/voice_extractor_weights.pth` | 439552 | `bf6bb1432dfa2a7c4cbed10aab1492316fdb89c2f9d7ff7e0423de33d5ea0b1e` | PyTorch state_dict | voice extractor (pytorch backend) | REBUILT by Phase 0, parameter-identical |

## 2. Which files does the runtime actually use?

`configs/default.yaml` selects exactly three graphs, all INT8:

```
models.face.onnx   = face_extractor_quant.onnx
models.voice.onnx  = voice_extractor_quant.onnx
models.fusion.onnx = attention_fusion_quant.onnx
```

The `*_fp32.onnx` files are **not** used by default. They are shipped because
they are the lossless source the Phase 0 recovery read the original weights
from, and because `scripts/benchmark.py` uses them as the fp32 reference point
for the INT8 comparison.

The `.pth` checkpoints are only read by `--backend pytorch` (needed to expose
the attention-gate weights), by `scripts/export.py`, and read-only by
`scripts/train/train_fusion.py`.

## 3. Classification

| class | files | meaning |
| --- | --- | --- |
| canonical | the three `*_quant.onnx` | what the shipped pipeline runs; do not replace silently |
| reference | the three `*_fp32.onnx` | untouched originals, kept for rebuild/benchmark |
| reconstructed | the three `.pth` | rebuilt after the Phase 0 incident, verified against the untouched fp32 graphs |

Reconstruction quality (from `_phase0_audit/stepB_report.json`):

| model | recovery | max abs difference vs the untouched graph |
| --- | --- | --- |
| `voice_extractor` | EXACT — all 8 tensors keep their original names | 4.53e-06 |
| `attention_fusion` | EXACT — 16/16 tensors (`classifier.1.num_batches_tracked` zero-filled, unused in eval) | 1.79e-07 |
| `face_extractor` | FUNCTIONALLY EQUIVALENT — the exporter folded all four `BatchNorm2d` layers into the preceding `Conv`, so the original BN statistics are not recoverable; BN was set to identity | 5.36e-07 |

## 4. Known residual risk

* `face_extractor_weights.pth` is behaviour-identical to the original but not
  byte-identical, and its BatchNorm running statistics are gone (the folded
  Conv weights carry the same composed function, which is why the embedding
  difference stays at 5.4e-07).
* `voice_extractor_weights.pth` and `attention_fusion_weights.pth` are
  parameter-identical; the zip/pickle container bytes may differ from the
  2026-04-19 files.
* If the pre-incident originals are needed byte-for-byte, they may still exist
  in the OneDrive version history of `weights/pytorch_pth/`. After restoring
  them, re-run `_phase0_audit/verify_inplace_weights.py`; the genuine mean must
  remain `0.9182`, and this manifest must be regenerated.

## 5. Regenerating this manifest

```python
import hashlib, pathlib
root = pathlib.Path("weights")
for path in sorted(root.rglob("*")):
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print("| `%s` | %d | `%s` |" % (path.as_posix(), path.stat().st_size, digest))
```

`/` separators are used for portability; on Windows the file lives at
`weights\onnx\...`.

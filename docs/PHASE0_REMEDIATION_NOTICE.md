# Phase 0 Remediation Notice — accidental re-training of 3 checkpoints

> **Read this first.** During Phase 0 read-only verification three files were
> modified by accident. They have been restored from the untouched ONNX graphs.
> One of the three could only be restored *functionally*, not byte-for-byte.
> You must know this before publishing anything.

## 1. What was executed

While looking for the project's entry point, `scripts_pc/*.py` were launched with
a generic interpreter to record their behaviour (a normal read-only audit step
for scripts that only *read*). Three of those scripts are **not** read-only:

| Script | Writes to | Executed? |
| --- | --- | --- |
| `scripts_pc/01_process_audio.py` | `data/processed/voice_features/` | no — aborted (`librosa` missing) |
| `scripts_pc/02_process_face.py` | `data/processed/face_features/` | started, **crashed before the first write** (Haar-cascade path) |
| `scripts_pc/03_train_single_models.py` | `weights/pytorch_pth/{voice,face}_extractor_weights.pth` | **yes — completed, overwrote** |
| `scripts_pc/04_train_fusion_scheme_b.py` | `weights/pytorch_pth/attention_fusion_weights.pth` | **yes — completed, overwrote** |
| `scripts_pc/05_export_and_quantize.py` | `weights/onnx_int8/` | no — aborted (`onnxruntime` missing) |

## 2. Exact damage (verified by SHA-256 diff of full inventories)

`_phase0_audit/diff_inventory.py` compares
`phase0_inventory_BEFORE.csv` (taken *before* anything was executed) with
`phase0_inventory_FINAL.csv` (taken after remediation): 50 files added
(`docs/` + `_phase0_audit/`), 0 removed, and exactly 3 modified — the three
`.pth` files, which are now the **rebuilt** checkpoints.

The intermediate, post-incident state was captured in the same way and is quoted
verbatim below; it is also preserved on disk in
`_phase0_audit/retrained_by_phase0/` (the files created by the accident):

```
ADDED   : 0
REMOVED : 0
MODIFIED: 3
   ~ Pi_Multimodal_Auth_Project/weights/pytorch_pth/attention_fusion_weights.pth
       size 115319 -> 114888   mtime 2026-04-19T14:59:22 -> <audit time>
   ~ Pi_Multimodal_Auth_Project/weights/pytorch_pth/face_extractor_weights.pth
       size 1703845 -> 1702970 mtime 2026-04-19T14:59:15 -> <audit time>
   ~ Pi_Multimodal_Auth_Project/weights/pytorch_pth/voice_extractor_weights.pth
       size 440749 -> 440256   mtime 2026-04-19T14:59:15 -> <audit time>
```

No dataset, feature, source, config or ONNX file changed.
**The six `weights/onnx_int8/*.onnx` files were not touched** (mtime still
2026-04-19 14:59:28), which is what makes recovery possible.

## 3. Recovery

Original weights still exist inside the fp32 ONNX graphs, because
`scripts_pc/05_export_and_quantize.py` exported them from the original `.pth`.
`_phase0_audit/stepA1_onnx_dump.py` + `stepB_rebuild_weights.py` extracted them
and rebuilt the checkpoints.

Verification of the rebuilt checkpoints against the untouched ONNX graphs
(`_phase0_audit/stepB_report.json`):

| comparison | max abs difference |
| --- | --- |
| voice extractor embeddings | 4.53e-06 |
| face extractor embeddings | 5.36e-07 |
| fusion score | 1.79e-07 |

| model | recovery quality |
| --- | --- |
| `voice_extractor` | **EXACT** — all 8 tensors keep their original PyTorch names; the graph has no BatchNorm so nothing was folded. |
| `attention_fusion` | **EXACT** — 16/16 tensors. Only `classifier.1.num_batches_tracked` (an unused step counter in eval mode) was zero-filled. |
| `face_extractor` | **FUNCTIONALLY EQUIVALENT** — the ONNX exporter folded all four `BatchNorm2d` layers into the preceding `Conv`, so the original BN `gamma/beta/running_mean/running_var` are *not* recoverable. The folded Conv weights were taken verbatim and the BN layers were set to identity (`gamma=1, beta=0, mean=0, var=1-eps`), which reproduces the original composed function to 5.4e-07. |

The restored files are now in place at
`Pi_Multimodal_Auth_Project/weights/pytorch_pth/`.
The re-trained versions produced by the accident are preserved for full
traceability in `_phase0_audit/retrained_by_phase0/`.

## 4. Residual risk and your options

* `voice_extractor_weights.pth` and `attention_fusion_weights.pth` are
  parameter-identical to the originals; only pickle/zip serialisation bytes may
  differ from the 2026-04-19 files.
* `face_extractor_weights.pth` is behaviour-identical but **not byte-identical**
  and its BN statistics are lost. If a byte-exact copy matters, use OneDrive
  version history: right-click the file → *Version history* → pick the
  2026-04-19 version. The folder lives inside OneDrive, so the old revisions
  should still be available there.
* If you restore from OneDrive, re-run
  `_phase0_audit/verify_inplace_weights.py` and compare with
  `_phase0_audit/inplace_verification.json` — the numbers must stay
  `genuine mean = 0.9182`.

## 5. Process fix for the rest of the project

* Treat `scripts_pc/0x_*.py` as **write operations**; never run them during an
  audit.
* Keep the whole `_phase0_audit/` folder (scripts + JSON evidence) until Phase 1
  ends; it is the only record of the pre-audit state.

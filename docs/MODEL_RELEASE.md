# Model release record

Companion to `MODEL_NOTICE.md` (the terms) and
`docs/MODEL_ARTIFACT_MANIFEST.md` (size + SHA-256 of every file). This document
records **what is published, why, and under which rights**.

## 1. Decision

| question | answer |
| --- | --- |
| Are the trained parameters published? | **Yes** — all nine files under `weights/` |
| Under the MIT code license? | **No** — separate terms in `MODEL_NOTICE.md` |
| Those terms in one line | research, evaluation and non-commercial use; keep the notice and the attribution; retrain on your own data for commercial use |
| Reason for a separate notice | the networks were trained with the help of a research-use face database ( Olivetti/ORL), which cannot be relicensed here (see `docs/RELEASE_AUDIT.md` §5, §9) |
| Was any third-party pretrained model used? | **No** — the three networks are defined and trained in this repository |

## 2. What is published

Canonical runtime models (INT8, selected by `configs/default.yaml`):

| model | file | bytes | sha256 | params |
| --- | --- | --- | --- | --- |
| face encoder | `weights/onnx/face_extractor_quant.onnx` | 432865 | `0ba342ddf0b5ee345588985fbffcb4ea3a4164972d6df0c3bf99aaa7738e51a2` | 423,236 |
| voice encoder | `weights/onnx/voice_extractor_quant.onnx` | 117288 | `ec25946393150e25eadf6e8c263ec9215cc2e94c600d18273f76f2ec2425083b` | 109,184 |
| attention-gated fusion + head | `weights/onnx/attention_fusion_quant.onnx` | 40172 | `34c2c0586a69018d95b2a95d110bffce95e94cf911dcb15e10f33899b77d5bc0` | 27,140 |

Reference and research artifacts:

| model | file | bytes | sha256 |
| --- | --- | --- | --- |
| face encoder, fp32 ONNX | `weights/onnx/face_extractor_fp32.onnx` | 1687815 | `3949be0816c813a9da2b3cba82d1d601f62e4e12501d0d3eb5a1711126c76345` |
| voice encoder, fp32 ONNX | `weights/onnx/voice_extractor_fp32.onnx` | 438936 | `191697363be7eb0ad51b669922e8dd0caa6a6dccd500bc4437f8f7a575a25c8a` |
| fusion head, fp32 ONNX | `weights/onnx/attention_fusion_fp32.onnx` | 112106 | `d8272a3986bb773502d395ff628867c1100805397b0b75ba9619458f3233975c` |
| face encoder, PyTorch | `weights/pytorch/face_extractor_weights.pth` | 1701820 | `574fb82d9835f2bbc8f80617e92ac2e81814667fc360b2df9864b2c7dfcba6a7` |
| voice encoder, PyTorch | `weights/pytorch/voice_extractor_weights.pth` | 439552 | `bf6bb1432dfa2a7c4cbed10aab1492316fdb89c2f9d7ff7e0423de33d5ea0b1e` |
| fusion head, PyTorch | `weights/pytorch/attention_fusion_weights.pth` | 114019 | `a3145830354b5db772b30733ad7176338930db9b29cdc0a78fd11f3da2e40540` |

Total size: 4.85 MB. `tests/test_artifacts.py` fails if any byte of any of these
changes.

## 3. Interface contract

Shapes are read from the shipped ONNX graphs (batch axis is dynamic):

| graph | input | output |
| --- | --- | --- |
| `face_extractor_quant.onnx` | `face_input` `(batch, 3, 112, 112)` float32 | `output` `(batch, 128)` float32 |
| `voice_extractor_quant.onnx` | `voice_input` `(batch, 1, 40, 400)` float32 | `output` `(batch, 128)` float32 |
| `attention_fusion_quant.onnx` | `voice_input` `(batch, 128)`, `face_input` `(batch, 128)` float32 | `output` `(batch, 1)` in `[0, 1]` |

* Framework of record: PyTorch 2.7.1 (`state_dict`s), exported to ONNX opset 11,
  then dynamically quantised to INT8 by `scripts/export.py`.
* Runtime: `onnxruntime==1.24.1` (the INT8 graphs contain `ConvInteger`, which
  older runtimes cannot parse).
* Input preprocessing: Haar-cascade detection → 112×112 crop → `(x-127.5)/128`
  for faces; 16 kHz mono → 40 MFCC → 400-frame crop/zero-pad for voice. See
  `configs/default.yaml`.

## 4. Training origin

| model | data | procedure |
| --- | --- | --- |
| `face_extractor` | the author's 5 face photos (positives) + 50 Olivetti/ORL images (negatives) | `scripts_pc/03_train_single_models.py`, 20 epochs, **no seed** |
| `voice_extractor` | the author's 10 voice clips (positives) + 25 synthetic noise clips (negatives) | same script, 20 epochs, **no seed** |
| `attention_fusion` | embeddings of the above | `scripts_pc/04_train_fusion_scheme_b.py`, 200 steps |

The exact training runs are **not** reproducible (no seeds were recorded). The
shipped `.pth` files were rebuilt after the Phase 0 checkpoint incident from the
untouched fp32 ONNX graphs and verified to ≤5.4e-07; see
`docs/PHASE0_REMEDIATION_NOTICE.md` and `docs/MODEL_ARTIFACT_MANIFEST.md` §3.

## 5. Ownership and third-party attribution

* Copyright in the trained parameters: the project author. No third-party
  pretrained weights, and no third-party code, contributed to them.
* The Olivetti/ORL face images used as negatives remain the property of their
  publisher (AT&T Laboratories Cambridge) and are **not** redistributed here.
  Required credit: "AT&T Laboratories Cambridge" — see
  `THIRD_PARTY_NOTICES.md` §3.1.
* The synthetic noise clips and the author's own enrolment media are the author's.
* To reuse the training *recipe*, obtain the Olivetti/ORL database yourself and
  observe its research-use terms.

## 6. What a user may and may not do

| allowed | forbidden without separate permission |
| --- | --- |
| run the models locally, in research, in teaching, in evaluation | commercial deployment |
| study, benchmark and publish results about them, with attribution | redistributing them without `MODEL_NOTICE.md` |
| modify and retrain them | presenting the system as certified or production-grade biometric security |
| redistribute them with the notice intact | using them on other people's biometric data without consent |

## 7. Commercial use

If a commercial deployment is ever intended: retrain the networks on data you have
the rights to (`scripts/train/*` reproduces the recipe) and delete the shipped
parameters. That removes the derived-from-research-data question entirely
(`docs/RELEASE_AUDIT.md` §10.6).

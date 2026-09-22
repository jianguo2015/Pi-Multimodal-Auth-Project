# PHASE0_AUDIT_REPORT.md — final Phase 0 audit result

Date: 2026-09-22 · Scope: `C:\Users\陶兴宇\OneDrive\Desktop\CatDog\CatDog`
Mode: evidence-first, read-only (with one accidental write, fully remediated —
see `PHASE0_REMEDIATION_NOTICE.md`).

---

## ⚠️ 0. Incident first

`scripts_pc/03` and `scripts_pc/04` are **not** read-only; running them during
verification re-trained and overwrote the three `.pth` checkpoints. This was
detected with a SHA-256 before/after inventory diff, and the original weights were
rebuilt from the untouched fp32 ONNX graphs and verified to 5.4e-07.
Details and residuals: `PHASE0_REMEDIATION_NOTICE.md`.

---

## A. Project Identity

* **What it is**: a two-modality (face image + voice MFCC) identity-verification
  prototype. Each modality gets a hand-written tiny CNN that outputs a 128-d
  embedding; an attention-gated fusion MLP turns the pair into a single
  "same person" score; a threshold at 0.85 yields ACCEPT/REJECT. Training is done
  on a PC and exported to ONNX fp32 + INT8 for a Raspberry-Pi-class target.
* **What it solves**: demonstrating a complete, small-footprint multimodal
  biometric pipeline that fits on an edge device (560 K parameters, <5 MB of
  weights in total).
* **What it is not**: not a security product. There is no real capture, no
  liveness check, no evaluation harness, and no Pi measurement.

## B. Technical Stack

Python 3.13.1 (`D:\Python313`) · PyTorch 2.7.1+cu118 · torchvision · ONNX 1.20.1 ·
ONNX Runtime 1.24.1 (Pi side pins 1.16.0 → broken, see D/F) · OpenCV 4.13 ·
librosa 0.11.0 · scikit-learn 1.8.0 · numpy 2.3.5 · soundfile · psutil · PyYAML ·
sounddevice (missing in the original env).
**Not used anywhere**: NCNN, OpenVINO, TFLite, TensorRT, RKNN, TensorFlow.
No Git repository, no CI, no tests, no README, no LICENSE.

## C. Architecture

```
raw media ─▶ preprocessing (cv2 / librosa) ─▶ 128-d emb (voice)
                                             128-d emb (face) ─▶ attention fusion
                                                              ─▶ sigmoid score
                                                              ─▶ threshold 0.85
                                                              ─▶ ACCEPT / REJECT
```
Offline: `scripts_pc/00..05` produce `data/processed/*.npy` →
`weights/pytorch_pth/*.pth` → `weights/onnx_int8/*.onnx`.
Edge: `app_pi/pi_main.py` loads the three `*_quant.onnx` and runs the decision.
Full diagrams: `CORE_ARCHITECTURE.md`.

## D. Core Code

`models/voice_extractor.py`, `models/face_extractor.py`,
`models/attention_fusion.py`, `utils/audio_ops.py`, `utils/vision_ops.py`,
`config/settings.yaml`, `scripts_pc/01..05`, `app_pi/pi_main.py`,
`app_pi/hardware_monitor.py`. 18 `.py` files / 48 KB total; 13 belong to the
project. Everything else in the folder is coursework or junk
(`train_gan.py`, `1.py`, `2.py`, `3.py`, `my_gan_data/`, `pokemon.zip`).

## E. Model System

| model | input | output | params | ONNX fp32 / INT8 |
| --- | --- | --- | --- | --- |
| VoiceExtractor | (B,1,40,400) f32 | (B,128) | 109,184 | 438,936 B / 117,288 B |
| FaceExtractor | (B,3,112,112) f32 | (B,128) | 423,236 | 1,687,815 B / 432,865 B |
| ModalAttentionFusion | (B,128)+(B,128) | (B,1) ∈ (0,1) | 27,140 | 112,106 B / 40,172 B |

Training: closed-set 6-class classification for the two extractors (head discarded
afterwards); 200 hand-built 4-sample steps with BCE for the fusion head.

## F. Multimodal Fusion

**Late, embedding-level, learned attention-gated fusion**: two linear projections
to 64-d, a softmax gate over the concatenation, a convex combination
(`w_v·v + w_f·f`), then an MLP classifier ending in a sigmoid. Not early fusion,
not score fusion. Measured INT8-vs-fp32 agreement: genuine score 0.9198 vs 0.9182.

## G. Current Reproduction

| capability | status |
| --- | --- |
| face preprocessing, bit-identical (55/55) | `VERIFIED` |
| voice preprocessing, bit-identical (35/35) | `VERIFIED` |
| fp32 ONNX end-to-end | `VERIFIED` (ORT 1.17.3 / 1.20.1 / 1.24.1) |
| INT8 ONNX end-to-end | `VERIFIED` on ORT 1.24.1; **fails** on ≤1.20.1 (`ConvInteger` unimplemented) |
| the real edge entry `pi_main.py` | `VERIFIED` to run on ORT 1.24.1; **fails** with the pinned ORT 1.16.0 |
| full decision path on real features | `VERIFIED`: genuine 0.9182 [0.8943, 0.9293] accept@0.85 = 1.00; impostors ≤ 0.0483, accept@0.85 = 0.00 |
| latency on this x86 laptop | PyTorch 2.02 ms · ONNX fp32 1.48 ms · **ONNX INT8 33.42 ms** |
| training scripts | executable (they overwrite checkpoints) |
| real capture / Pi run / accuracy metrics | **not reproducible** |

## H. Historical Edge Deployment

* **LEVEL A** — the PC-side pipeline really ran: `.pth` and `.onnx` mtimes
  `2026-04-19 14:59:15 … 14:59:28`, plus PyCharm coverage entries in
  `.idea/workspace.xml` for all six scripts between 14:59:00 and 14:59:25.
* **LEVEL B** — complete edge code + dependency list exist, and the edge code was
  proven to run end-to-end on this PC with ORT 1.24.1 (score 0.0230 from random
  inputs). But there is **no** Pi log, screenshot, deployment script, systemd unit
  or performance number anywhere.
* **LEVEL C** — every "Raspberry Pi performance" claim. The only report the code
  can produce is fed by `np.random.randn` and reports `temp_c = 0.0` off-Pi.

> Historical deployment evidence only. Current hardware reproduction unavailable.

## I. Evidence Quality (per claim)

| claim | evidence | status |
| --- | --- | --- |
| 双模态（人脸+声纹）身份验证 | `models/*`, `pi_main.py`, score matrix | `VERIFIED` |
| 模型可推理 | `.pth` and `.onnx` executed on real features | `VERIFIED` |
| 多模态融合（注意力门控） | `attention_fusion.py` + ONNX `Softmax`/`Mul`/`Add` | `VERIFIED` |
| 预处理可复现 | 55/55 + 35/35 bit-identical | `VERIFIED` |
| ONNX 导出（fp32） | 6 files, all load & run | `VERIFIED` |
| INT8 量化导出 | `quantize_dynamic(QInt8)` output loads on ORT 1.24.1 | `VERIFIED` |
| INT8 精度保持 | 0.9198 vs 0.9182 | `VERIFIED` (toy data) |
| INT8 加速 | measured **22× slower** on x86 | **REFUTED** |
| 树莓派部署 | `app_pi/` + `requirements_pi.txt` only | `HISTORICAL` |
| 树莓派性能数据 | nothing exists | `UNVERIFIED` |
| 真实采集 | `dummy_capture()` only | **REFUTED** |
| 训练脚本可运行 | executed during this audit | `VERIFIED` |
| 训练可复现 | no seeds | `UNVERIFIED` |
| 精度 / FAR / FRR / EER | no harness; data too small | `UNVERIFIED` |
| NCNN / OpenVINO / TFLite 部署 | zero hits | **REFUTED** |
| 活体 / 防伪 | not implemented | `UNVERIFIED` |
| Git 历史 | no `.git` | **N/A** |

## J. Technical Debt (top items)

1. `requirements_pi.txt onnxruntime==1.16.0` cannot load the shipped INT8 models → deployment blocker.
2. No capture layer; the edge entry is a demo harness.
3. Emoji print crashes on non-UTF-8 consoles and is swallowed as "init failed".
4. `00_download_face_data.py` would `rmtree` the real user photos.
5. Config drift (`epochs_fusion`, `fusion_hidden_dim`) + 5× duplicated config loading.
6. No seeds, no validation split, no metrics, no logging, no tests, no README, no LICENSE.
7. All scripts are CWD-dependent; `cv2.imread` fails on non-ASCII absolute paths.
8. The project's own `.venv` is broken; the real environment lives outside the repo.

## K. Open Source Plan

* **Publish**: `app_pi/`, `models/`, `utils/`, `config/`, `scripts_pc/01..05`
  (drop `00_download_face_data.py`), `weights/*` (≈5 MB), plus new
  `README.md` / `LICENSE` / `.gitignore` / `docs/` / a synthetic-data generator.
* **Document as history**: the Raspberry Pi deployment attempt — in `docs/`,
  explicitly labelled *historical evidence only*.
* **Never publish**: `data/raw/{face,voice}/User_Me/`,
  `data/processed/**/User_Me/`, `live_capture.jpg`, `.idea/`, `.vs/`,
  `.venv*`, `my_gan_data/`, `pokemon.zip`, `1.py`–`3.py`, `train_gan.py`,
  `data/cifar-10*`. Full detail: `OPEN_SOURCE_RISK.md`.

## L. Refactoring Plan (Phase 1 — not started)

1. Fix the ORT requirement / quantization scheme so the INT8 path is loadable and
   genuinely faster (or drop the INT8 speedup claim entirely).
2. Add `capture.py` reusing the exact `utils/` preprocessing functions.
3. Single entry points (`run_pc.py`, `run_pi.py`) + `argparse` + `__file__`-relative paths.
4. Central `config.py` with validation; delete the five duplicated YAML reads.
5. `evaluate.py` producing threshold sweeps, confusion counts and FAR/FRR with an
   explicit "toy data" warning; turn Phase 0's measurements into a regression baseline.
6. Seeds + `--deterministic` mode; persist metrics with the checkpoints.
7. Split the repository: keep the biometric pipeline, move the coursework out.
8. Engineering basics: README, LICENSE, tests, Makefile/nox, CI.

## M. Resume Claims

**Safe to claim (all `VERIFIED`)**

* Built a from-scratch two-modality (face + voice) biometric verification pipeline
  with a custom attention-gated fusion network (~560 K parameters, <5 MB weights).
* Implemented the whole PC pipeline — dataset bootstrap → feature extraction →
  single-modality training → fusion training → ONNX export → INT8 dynamic
  quantization — with bit-for-bit reproducible preprocessing (55/55 faces, 35/35 voices).
* Validated the INT8 export against the fp32 graph: genuine-score agreement
  0.9198 vs 0.9182 across all 50 genuine pairs, and complete genuine/impostor
  separation on the prototype dataset (0.9182 vs ≤0.0483).
* Delivered an end-to-end inference artefact of ~570 KB (INT8 ONNX) with a
  measured ~1.3–1.5 ms CPU forward pass on a laptop.

**Must be stated carefully — or not at all**

* Raspberry Pi: write *"prototype ported to a Raspberry-Pi-targeted ONNX Runtime
  deployment (historical, not re-verified — hardware no longer available)"*.
  Never write "deployed and optimised on Raspberry Pi at X FPS".
* Quantization: do **not** claim a speedup — measured INT8 33.42 ms vs fp32
  1.48 ms on x86 CPU, because `ConvInteger` has no optimised CPU kernel.
* Accuracy: do **not** quote accuracy/F1/FAR/FRR — the negative-class voice data
  is synthetic noise and there is exactly one genuine identity.
* Never quote any Pi temperature / CPU / RAM / FPS number: none exists.


---

## N. Fact–Evidence Matrix

| Claim | Evidence | Status |
| --- | --- | --- |
| 多模态身份验证（人脸+声纹） | `models/*.py`, `app_pi/pi_main.py` L47-62, score matrix | **VERIFIED** |
| 注意力门控融合 | `models/attention_fusion.py`, ONNX graph nodes | **VERIFIED** |
| 模型推理（PyTorch 权重） | `verify_inplace_weights.py`: 50 genuine + 35 impostor combos | **VERIFIED** |
| 模型推理（ONNX fp32） | `stepA2_onnx_run.py` on ORT 1.17.3 / 1.20.1 / 1.24.1 | **VERIFIED** |
| 模型推理（ONNX INT8） | same script, ORT 1.24.1 only | **VERIFIED** (version-dependent) |
| ONNX 部署（PC 侧导出） | 6 `.onnx`, mtime 2026-04-19 14:59:28 | **VERIFIED** |
| ONNX Runtime 作为边缘运行时 | `pi_main.py` L4, L23-25; `requirements_pi.txt` | **VERIFIED** (as code) |
| INT8 量化已被执行 | `05_export_and_quantize.py` + `_quant.onnx` present | **VERIFIED** |
| INT8 加速 | measured 33.42 ms vs 1.48 ms | **REFUTED** |
| Raspberry Pi 部署 | `app_pi/` + `requirements_pi.txt` + Pi telemetry code | **HISTORICAL** |
| Raspberry Pi 性能数据 | none in repo; report fed by random input | **UNVERIFIED** |
| 真实摄像头/麦克风采集 | `dummy_capture()` returns `np.random.randn` | **REFUTED** |
| NCNN 部署 | zero hits | **REFUTED** |
| OpenVINO 部署 | zero hits | **REFUTED** |
| TFLite / TensorRT / RKNN 部署 | zero hits | **REFUTED** |
| 性能数据（FPS / Accuracy / F1 / 延迟） | only Phase 0's own x86 measurements exist | **VERIFIED** for §G numbers only; everything else **UNVERIFIED** |
| 数据集为真实多人语音 | stranger voice is synthetic noise | **REFUTED** |
| 训练可复现 | no seeds anywhere | **UNVERIFIED** |
| 代码经过测试 | no tests exist | **REFUTED** |
| 版本管理 | no `.git` | **N/A** |
| 项目名 "CatDog" 与内容是否相符 | 目录名为 CatDog，但代码、模型、数据均无任何猫狗相关逻辑；`CatDog` 应视为仓库/课程代号 | **UNKNOWN** — 需作者确认命名来源 |

## O. Phase 0 Stop Condition

```
[x] 我知道所有核心目录
[x] 我知道真正入口（app_pi/pi_main.py；scripts_pc/00..05）
[x] 我知道真正调用链（逐函数，见 PROJECT_CONTEXT §6）
[x] 我知道所有主要模型（3 个，参数为实测值）
[x] 我知道模型输入输出（ONNX 输入名与形状为实测值）
[x] 我知道多模态在哪里融合（attention_fusion.py L31-44，late attention-gated）
[x] 我知道决策逻辑（单阈值 0.85）
[x] 我知道当前如何运行（用 D:\Python313；见 REPRODUCTION_STATUS.md）
[x] 我知道哪些代码是旧实验（00_download_face_data.py, 1/2/3.py, train_gan.py, my_gan_data）
[x] 我知道哪些代码是真正核心（models/, utils/, scripts_pc/01..05, app_pi/）
[x] 我知道哪些资源不能公开（User_Me 原始媒体与特征、live_capture.jpg、.idea）
[x] 我知道 Raspberry Pi 历史证据在哪里（app_pi/ + requirements_pi.txt + 2026-04-19 mtimes + .idea coverage 时间戳）
[x] 我知道当前哪些功能可以复现（预处理、ONNX fp32/INT8、PyTorch 判决、训练脚本）
[x] 我知道当前哪些功能无法复现（Pi 实机、真实采集、原始随机权重、任何精度指标）
[x] 我知道项目真正值得开源的工程价值（端到端可复现的双模态流水线 + 注意力门控融合 + ONNX/INT8 导出与量化一致性验证）
[x] 我有完整 PROJECT_CONTEXT.md
```

## P. Deliverables produced by Phase 0

| file | content |
| --- | --- |
| `docs/PROJECT_CONTEXT.md` | 25-section project cognition document |
| `docs/CORE_ARCHITECTURE.md` | architecture / data / model / fusion / decision flow + failure domains |
| `docs/CODEBASE_MAP.md` | per-file responsibilities, callers, status |
| `docs/REPRODUCTION_STATUS.md` | REPRODUCIBLE / PARTIALLY / HISTORICAL / UNVERIFIED |
| `docs/OPEN_SOURCE_RISK.md` | privacy, licensing, hygiene, publication checklist |
| `docs/PHASE0_AUDIT_REPORT.md` | this report (incl. Fact–Evidence matrix) |
| `docs/PHASE0_REMEDIATION_NOTICE.md` | the accidental checkpoint overwrite and its recovery |
| `_phase0_audit/phase0_tree.txt` | ASCII tree of the whole CatDog folder |
| `_phase0_audit/phase0_inventory.csv` | 619 files × sha256 / size / mtime / type / lines / encoding / image dims |
| `_phase0_audit/phase0_inventory_BEFORE.csv`, `..._FINAL.csv`, `diff_inventory.py` | the before/after proof of the incident and of the remediation |
| `_phase0_audit/onnx_dump/` | all 34 ONNX initializers used to rebuild the checkpoints |
| `_phase0_audit/preprocessing_probe.json` | 55/55 + 35/35 bit-identical evidence |
| `_phase0_audit/onnx_outputs.npz` (+ `_ort1201.npz`) | real ONNX embeddings, score grids, loadability matrix, latencies |
| `_phase0_audit/stepB_report.json` | checkpoint reconstruction + verification vs ONNX |
| `_phase0_audit/inplace_verification.json` | score matrix from the in-place `.pth` |
| `_phase0_audit/restored_original/` + `retrained_by_phase0/` | recovered checkpoints and the accidental ones |
| `_phase0_audit/probe_media.py` / `probe_media_result.json` | EXIF / WAV / npy-shape evidence |
| `_phase0_audit/run_pi_entry_harness.py` + `_pi_harness.txt` | the real edge entry executed unmodified |
| `_phase0_audit/_run_pi_main.txt` | the pinned-environment failure reproduction |
| `_phase0_audit/_edge_scan.txt` | Raspberry-Pi / ONNX keyword scan results |

### Git

```
$ git status / git log / git remote -v
fatal: not a git repository (or any of the parent directories): .git
```
No `.git` exists at `CatDog\` nor at `CatDog\CatDog\Pi_Multimodal_Auth_Project\`,
so there is no branch, commit or remote history to audit. Nothing was
initialised (Phase 0 must not create one).

---

**Phase 0 结束。** 未重构、未新建架构、未修改业务代码、未发布 GitHub。
除 `docs/`（本次交付文档）与 `_phase0_audit/`（审计产物）之外，唯一的写入是
`weights/pytorch_pth/*.pth` 的误改与已完成的恢复 —— 见 `PHASE0_REMEDIATION_NOTICE.md`。
等待下一阶段任务。


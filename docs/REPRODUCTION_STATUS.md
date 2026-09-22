# REPRODUCTION_STATUS.md — CatDog / Pi_Multimodal_Auth_Project

What can actually be re-run on this Windows machine today, what needs repair, what
exists only as history, and what was never real.
All commands below were executed during Phase 0 unless marked otherwise.

---

## CURRENTLY_REPRODUCIBLE

### 1. Face preprocessing — bit-identical ✔
```powershell
<original-interpreter>\python.exe _phase0_audit\probe_preprocessing.py
# face_total: 55, face_identical: 55
```
`_phase0_audit/preprocessing_probe.json`. The probe calls the project's own
`utils/vision_ops.extract_face` from the same CWD (`scripts_pc/`) with the same
relative paths, then compares against the stored `.npy` using `np.array_equal`
→ **55/55 identical**.

### 2. Voice preprocessing — bit-identical ✔
Same run: `voice_total: 35, voice_identical: 35`.
Caveat: only holds with **librosa 0.11.0** (the original environment); another
librosa version changes the resampler and breaks byte equality.

### 3. fp32 ONNX end-to-end inference ✔
```powershell
<original-interpreter>\python.exe _phase0_audit\stepA2_onnx_run.py
# [fp32] v_emb(10,128) f_emb(5,128) grid(10,5) mean=0.9182 min=0.8943 max=0.9293
# [fp32] end-to-end latency (CPU, this host): 1.48 ms/sample
```
Works on ORT 1.17.3, 1.20.1 and 1.24.1.

### 4. INT8 ONNX end-to-end inference ✔ (needs a recent ONNX Runtime)
Same command, INT8 branch, on **ORT 1.24.1** (the original interpreter):
```
[quant] mean=0.9198 min=0.8945 max=0.9328
[quant] end-to-end latency (CPU, this host): 33.42 ms/sample
[load] voice_extractor_quant.onnx  LOADED
[load] face_extractor_quant.onnx   LOADED
[load] attention_fusion_quant.onnx LOADED
```

### 5. The real edge entry point — runs, and exposes two defects ✔
```powershell
<original-interpreter>\python.exe _phase0_audit\run_pi_entry_harness.py
```
```
[Pi] 初始化多模态推理引擎...
=== 开始基于树莓派的多模态身份认证 ===
-> 融合置信度: 0.0230 (阈值: 0.85)
初始化失败: 'gbk' codec can't encode character '\U0001f534' ...
```
Reading: (a) with ORT 1.24.1 the three INT8 models **do** load and the full
voice→face→fusion→threshold path executes; (b) the score comes from
`np.random.randn`, so it means nothing; (c) the verdict print crashes on a
non-UTF-8 console. The harness injects a stub only for the one missing import
(`sounddevice`); `pi_main.py` itself is untouched.

### 6. The failure mode of the pinned Pi environment ✔ (reproduced twice)
```powershell
# with onnxruntime 1.20.1 (isolated audit environment)
cd Pi_Multimodal_Auth_Project\app_pi ; python pi_main.py
# -> 初始化失败: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED :
#    Could not find an implementation for ConvInteger(10) node with name '/conv1/Conv_quant'
```
`pi_main.py` + `requirements_pi.txt` as shipped **cannot initialise**;
`pi_main.py` + ORT ≥ 1.24.1 works. The mismatch is the requirement pin, not the
model files.

### 7. PyTorch weights, full score matrix and decision threshold ✔
```powershell
<other-interpreter>\python.exe _phase0_audit\verify_inplace_weights.py
```
| pair | n | mean | min | max | accept@0.85 |
| --- | --- | --- | --- | --- | --- |
| User_Me voice + User_Me face (**genuine**) | 50 | **0.9182** | 0.8943 | 0.9293 | **1.00** |
| all 35 impostor combinations | 25–100 each | 0.0045 – 0.0251 | 0.0042 | **0.0483** | **0.00** |

Stored in `_phase0_audit/inplace_verification.json`.
**Honest limits**: the genuine side is 10 voice × 5 face samples of one person;
impostor voice is **synthetic noise**; impostor faces come from the public Olivetti
dataset. This shows the pipeline works end to end — it is **not** a biometric
benchmark and must never be quoted as accuracy/FAR/FRR.

### 8. Training scripts execute ✔ (but they overwrite the checkpoints)
`scripts_pc/03_train_single_models.py` and `04_train_fusion_scheme_b.py` were
executed during Phase 0 and completed (20 epochs × 2 extractors; 200 fusion steps).
They are functional — and they caused the checkpoint overwrite documented in
`PHASE0_REMEDIATION_NOTICE.md`. **Never run them for an audit.**


---

## PARTIALLY_REPRODUCIBLE

| capability | what is missing | how to fix |
| --- | --- | --- |
| `scripts_pc/01_process_audio.py` as a script | `librosa` in the interpreter used | use the original interpreter (librosa 0.11.0 present) |
| `scripts_pc/02_process_face.py` as a script | must run from `scripts_pc/`; `cv2.data.haarcascades` must sit on an ASCII path | use the original interpreter (ASCII install dir) |
| `scripts_pc/05_export_and_quantize.py` | needs `onnx` + `onnxruntime.quantization` in one process | present in the original interpreter (onnx 1.20.1) |
| running the edge entry without a harness | `sounddevice` is not installed in the original interpreter | `pip install sounddevice==0.4.6` |
| INT8 inference on this laptop | an ONNX Runtime version that implements `ConvInteger` | ORT 1.24.1 works; 1.17.3 / 1.20.1 do not |

---

## HISTORICAL_ONLY

| item | evidence that it happened | why it cannot be re-verified |
| --- | --- | --- |
| **Raspberry Pi deployment** | `app_pi/` code + `requirements_pi.txt` + Pi-only telemetry code + INT8 models | *Historical deployment evidence only. Current hardware reproduction unavailable.* No Pi, no Pi log, no Pi screenshot, no Pi performance number anywhere in the repo. |
| The original training run | `weights/pytorch_pth/*.pth` mtime 2026-04-19 14:59:15/22, ONNX 14:59:28, PyCharm coverage entries 14:59:00–14:59:25 in `.idea/workspace.xml` | no seeds → the exact weights cannot be regenerated; the original `face_extractor` BN statistics are unrecoverable (folded into ONNX) |
| The original interpreter state | `.idea/misc.xml` SDK "Python 3.13" + `__pycache__/*.cpython-313.pyc`; the original interpreter still holds torch 2.7.1+cu118 / ORT 1.24.1 | the project's own `.venv` is broken (no numpy, no pip) and is *not* the environment that trained the models |
| `data/processed/*/User_Me/*.npy` provenance | the features are bit-reproducible today, so the generating pipeline is known | the `User_Me` **raw** media is the author's own face and voice; nothing else can confirm who was recorded |

---

## UNVERIFIED

| claim | why it stays UNVERIFIED |
| --- | --- |
| Any Raspberry Pi latency / FPS / CPU / RAM / temperature figure | no such number exists in the repo; the only numbers `pi_main.py` can print come from `np.random.randn` inputs and a `temp_c = 0.0` fallback |
| "INT8 quantization speeds up edge inference" | the opposite was measured on this host: 33.42 ms vs 1.48 ms (≈22× slower) with ORT 1.24.1 on CPU |
| "Real-time authentication" | no capture code, no latency-budget test |
| Accuracy / FAR / FRR / EER of the authentication system | cannot be computed from 10 genuine voice samples and noise-based impostors; no evaluation script exists |
| Anti-spoofing / liveness / replay resistance | not implemented, not mentioned in code |
| Any NCNN / OpenVINO / TFLite / TensorRT deployment | zero hits for those names anywhere in the repository |
| Multi-user enrolment (only one genuine identity exists) | `User_Me` is a single class |
| Git history / prior versions | no `.git` directory, no commits, no remote |

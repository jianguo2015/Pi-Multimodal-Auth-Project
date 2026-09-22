# CORE_ARCHITECTURE.md — CatDog / Pi_Multimodal_Auth_Project

Engineer's view: architecture, data flow, model flow, fusion flow, decision flow.
Everything is derived from the actual source files listed in `CODEBASE_MAP.md`.

## 1. System architecture (as built)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ OFFLINE (PC) — scripts_pc/00 .. 05           ┌────────────────────────┐  │
│                                              │ config/settings.yaml   │  │
│  00_setup_strangers.py ──┐                   │  (paths, dims, thr)    │  │
│  00_download_face_data.py│  (legacy)         └───────────┬────────────┘  │
│                          ▼                               │ read by all   │
│              data/raw/{face,voice}/<spk>/                │               │
│                          │                               │               │
│   01_process_audio.py ───┤ utils/audio_ops.extract_mfcc ─┤               │
│   02_process_face.py  ───┤ utils/vision_ops.extract_face ┤               │
│                          ▼                               │               │
│              data/processed/{voice,face}_features/<spk>/*.npy            │
│                          │                               │               │
│   03_train_single_models.py ──▶ models/{voice,face}_extractor.py         │
│                          ▼   weights/pytorch_pth/*_extractor_weights.pth │
│   04_train_fusion_scheme_b.py ─▶ models/attention_fusion.py              │
│                          ▼   weights/pytorch_pth/attention_fusion_*.pth  │
│   05_export_and_quantize.py ─▶ ONNX fp32 ─quantize_dynamic(QInt8)─▶ INT8 │
│                              weights/onnx_int8/*{_fp32,_quant}.onnx      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │ artifacts copied to the device
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ EDGE (Raspberry Pi) — app_pi/                                             │
│  pi_main.py :: PiMultimodalAuth                                           │
│      config  ← ../config/settings.yaml (relative to CWD, not __file__)    │
│      sessions ← onnxruntime.InferenceSession(_quant.onnx) x3 (2 threads)  │
│      monitor  ← hardware_monitor.PiHardwareMonitor (psutil + thermal_zone)│
│  authenticate(): capture → embed → fuse → threshold → print report        │
└──────────────────────────────────────────────────────────────────────────┘
        ▲
        └── NO real capture implementation: pi_main.dummy_capture() returns
            np.random.randn tensors (pi_main.py L29-36). Camera/microphone
            acquisition is described in a comment, never written.
```

## 2. Data flow

| stage | source | transform | artefact | verified |
| --- | --- | --- | --- | --- |
| face raw | 5 real photos (960×1280) + 50 Olivetti (112×112) | Haar detect → largest box → resize 112 → RGB → CHW → /128 | 55 × `.npy` (3,112,112) f32 | `VERIFIED` — 55/55 bit-identical regeneration |
| voice raw | 10 real recordings 48 kHz + 25 synthetic-noise 16 kHz/2 s | librosa load@16 kHz → MFCC 40 → pad/trunc 400 | 35 × `.npy` (1,40,400) f32 | `VERIFIED` — 35/35 bit-identical regeneration |
| labels | folder name | `FeatureDataset` sorts dir names → integer class | 6 classes: `Stranger_0..4`, `User_Me` | `VERIFIED` |
| fusion targets | `User_Me/*` vs everything else | 2 genuine + 2 mixed-modality impostor samples per step | 200 optimisation steps | `VERIFIED` (executed during audit) |

Latent train/serve skew: stored features are **already normalised**
(face ≈ [-1,1], MFCC ≈ [-93, 11]); no normalisation exists in the inference path,
so any future real-capture code must reproduce this exactly.

## 3. Model flow

```
(B,1,40,400) ─conv1(1→32,3×3)+ReLU+maxpool2 ─conv2(32→64,3×3)+ReLU+maxpool2
             ─conv3(64→128,3×3)+ReLU ─AdaptiveAvgPool(1,1) ─Linear(128→128)
             ──▶ (B,128) voice embedding          [109,184 params]

(B,3,112,112) ─Conv(3→32,s2)+BN+ReLU ─Conv(32→64,s2)+BN+ReLU
              ─Conv(64→128,s2)+BN+ReLU ─Conv(128→256,s2)+BN+ReLU
              ─AdaptiveAvgPool(1,1) ─Linear(256→128)
              ──▶ (B,128) face embedding          [423,236 params]
```

Both extractors emit into the **same 128-d space** but are trained independently
as 6-class closed-set classifiers; a temporary `nn.Linear(128, 6)` head is attached
for training and **discarded before saving** (`03_train_single_models.py` L92-94).


## 4. Fusion flow

```
voice_emb (B,128) ┐
                  ├─ v_proj Linear(128→64) ─ ReLU ─┐
face_emb  (B,128) ┘                              │ f_proj Linear(128→64) ─ ReLU
                                                  │
                       cat([v_feat, f_feat]) (B,128)
                                  │
                        Linear(128→64) ─ ReLU ─ Linear(64→2) ─ Softmax
                                  │
                    w = [w_v, w_f],  w_v + w_f = 1
                                  │
                fused = w_v * v_feat + w_f * f_feat      (B,64)
                                  │
        Linear(64→32) ─ BatchNorm1d(32) ─ ReLU ─ Dropout(0.2) ─ Linear(32→1)
                                  │
                              Sigmoid ──▶ score ∈ (0,1)        [27,140 params]
```

* **Late, embedding-level, learned attention-gated fusion.**
* `torch.cat` is used only to compute the gate; the fused representation is a
  convex combination of the two projected features (softmax weights sum to 1).
* `BatchNorm1d` is only meaningful in `eval()` mode. Both training and export
  keep eval behaviour, so this is a training-time fragility rather than an
  inference bug.
* Because there is no per-modality score, classic score-fusion metrics
  (per-modal FAR/FRR, per-modal AUC) **cannot** be measured from this
  architecture as written.

## 5. Decision flow

```
score     = fusion(...)[0][0][0]                    # pi_main.py L52
threshold = config.inference.fusion_threshold       # 0.85, settings.yaml L27
"认证通过" if score >= threshold else "拒绝访问"      # pi_main.py L62
```

No per-user calibration, no liveness / anti-spoofing, no capture-quality gating,
no retry policy, no rate limiting, no audit log.

## 6. Configuration flow

`config/settings.yaml` is the single source of truth and is re-read by five
scripts, always as `open("../config/settings.yaml")` — no shared loader,
no schema validation, no env override:

| key | used by | respected? |
| --- | --- | --- |
| `paths.*` | 01, 02, 03, 04, 05, `pi_main` | yes |
| `model.voice_n_mfcc`, `voice_max_pad_len`, `face_image_size`, `embed_dim` | 01–05, `pi_main` | yes |
| `model.fusion_hidden_dim: 64` | `pi_main` only | **no** — 04/05 rely on the class default |
| `train.batch_size`, `learning_rate`, `epochs_single` | 03 | yes |
| `train.epochs_fusion: 15` | — | **no** — 04 hardcodes `range(200)` |
| `inference.fusion_threshold` | `pi_main` | yes |

## 7. Failure domains

| # | failure | evidence |
| --- | --- | --- |
| F1 | INT8 `ConvInteger` graphs need a recent ONNX Runtime; the pinned Pi requirement cannot load them | `requirements_pi.txt onnxruntime==1.16.0` vs measured NOT_IMPLEMENTED on ORT 1.20.1 |
| F2 | Edge entry has no real data acquisition | `pi_main.py:29-36 dummy_capture()` |
| F3 | Emoji print raises on a GBK console | measured `UnicodeEncodeError` (`_phase0_audit/_pi_harness.txt`) |
| F4 | Everything depends on CWD | every script uses `"../…"`; `from hardware_monitor import …` |
| F5 | Non-ASCII absolute paths break `cv2.imread` | measured: absolute `C:\Users\陶兴宇\…` → None; relative path → OK |
| F6 | Training is not reproducible | no `torch.manual_seed` / `np.random.seed` anywhere |
| F7 | `00_download_face_data.py` deletes the real photos | `shutil.rmtree("../data/raw/face")` L12-13 |
| F8 | `.pth` and `_quant.onnx` may silently diverge | separate artefacts; nothing verifies they came from the same run |

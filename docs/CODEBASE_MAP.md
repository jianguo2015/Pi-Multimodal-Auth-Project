# CODEBASE_MAP.md — CatDog / Pi_Multimodal_Auth_Project

Every source file that exists, what it really does, and its status.
Status vocabulary: `CORE` · `ACTIVE` · `EXPERIMENT` · `LEGACY` · `DEMO` · `UNKNOWN`.
"deps" = runtime third-party imports only.

---

## app_pi/ — edge application

### `app_pi/pi_main.py` — 3,397 B — **status: DEMO** (edge entry point)
```
├── responsibility : the only end-to-end application. Loads 3 INT8 ONNX models,
│                    fakes a capture, runs embed→fuse→threshold, prints a report.
├── key classes    : PiMultimodalAuth
├── key functions  : __init__(config_path="../config/settings.yaml")
│                    dummy_capture()  -> (1,1,40,400) + (1,3,112,112) random f32
│                    authenticate()    -> voice emb, face emb, fused score, verdict
├── deps           : os, time, numpy, onnxruntime, cv2, sounddevice, yaml, hardware_monitor
├── callers        : __main__ only
├── called by      : nobody (top of the chain)
├── inputs         : ../config/settings.yaml, ../weights/onnx_int8/*_quant.onnx
├── outputs        : stdout only — no score, no log, no image is written
└── status reason  : the code path is real and measurably runs, but the data source is
                     np.random.randn while cv2/sounddevice are imported and never used
                     for capture -> it is a demonstration harness, not an auth system.
```

### `app_pi/hardware_monitor.py` — 890 B — **status: CORE (small utility)**
```
├── responsibility : Raspberry-Pi telemetry for the printed report.
├── key classes    : PiHardwareMonitor
├── key functions  : _check_if_pi()     -> reads /sys/firmware/devicetree/base/model
│                    get_cpu_temp()     -> /sys/class/thermal/thermal_zone0/temp /1000
│                    get_system_stats() -> {cpu_percent, ram_percent, ram_used_mb, temp_c}
├── deps           : psutil
├── callers        : pi_main.PiMultimodalAuth.__init__, .authenticate
└── notes          : off-Pi it silently returns temp_c = 0.0
```

---

## config/

### `config/settings.yaml` — 759 B — **status: CORE**
```
├── responsibility : the single configuration source
├── consumed by    : 01, 02, 03, 04, 05, pi_main  (each opens it independently)
├── keys           : paths{6}, model{voice_n_mfcc=40, voice_max_pad_len=400,
│                    face_image_size=112, embed_dim=128, fusion_hidden_dim=64},
│                    train{batch_size=32, learning_rate=0.001, epochs_single=20,
│                    epochs_fusion=15}, inference{fusion_threshold=0.85}
└── dead keys      : epochs_fusion (never read), fusion_hidden_dim (unused by 04/05)
```

---

## models/ — the project's actual contribution

### `models/voice_extractor.py` — 1,134 B — **status: CORE**
```
├── responsibility : MFCC image -> 128-d speaker embedding
├── key classes    : VoiceExtractor(n_mfcc=40, embed_dim=128)
├── key functions  : forward(x: (B,1,40,400)) -> (B,128)
├── architecture   : Conv(1→32,3×3)+ReLU+MaxPool2 · Conv(32→64,3×3)+ReLU+MaxPool2 ·
│                    Conv(64→128,3×3)+ReLU · AdaptiveAvgPool(1,1) · Linear(128→128)
├── params         : 109,184
├── deps           : torch, torch.nn, torch.nn.functional (F unused)
├── callers        : 03 (train), 04 (frozen extractor), 05 (ONNX export)
└── notes          : no BatchNorm — this is why its ONNX export is bit-exactly invertible
```

### `models/face_extractor.py` — 1,150 B — **status: CORE**
```
├── responsibility : 112×112 RGB face -> 128-d identity embedding
├── key classes    : FaceExtractor(embed_dim=128)
├── key functions  : forward(x: (B,3,112,112)) -> (B,128)
├── architecture   : 4 × [Conv 3→32→64→128→256, 3×3, stride 2, pad 1 + BatchNorm2d + ReLU]
│                    + AdaptiveAvgPool(1,1) + Linear(256→128)
├── params         : 423,236
├── deps           : torch, torch.nn, torch.nn.functional (F unused)
├── callers        : 03 (train), 04 (frozen), 05 (export)
└── notes          : comment "极简版 CNN，适应树莓派算力"; the 4 BN layers are folded into
                     Conv by torch.onnx.export(do_constant_folding=True)
```

### `models/attention_fusion.py` — 1,494 B — **status: CORE (the novelty)**
```
├── responsibility : attention-gated fusion of the two embeddings -> one confidence
├── key classes    : ModalAttentionFusion(voice_dim=128, face_dim=128, hidden_dim=64)
├── key functions  : forward(voice_emb (B,128), face_emb (B,128)) -> score (B,1)
├── architecture   : v_proj/f_proj Linear(128→64)+ReLU · attention_net
│                    [Linear(128→64), ReLU, Linear(64→2), Softmax] ·
│                    fused = w0*v_feat + w1*f_feat · classifier
│                    [Linear(64→32), BatchNorm1d(32), ReLU, Dropout(0.2), Linear(32→1), Sigmoid]
├── params         : 27,140
├── callers        : 04 (train), 05 (export), pi_main (inference)
└── notes          : torch.cat only feeds the gate; the fused vector is a convex
                     combination. Training script calls it "创新方案B".
```

---

## utils/ — preprocessing, shared by the PC scripts

### `utils/audio_ops.py` — 1,000 B — **status: CORE**
```
├── responsibility : raw wav -> fixed-shape MFCC tensor
├── key functions  : extract_mfcc(file_path, n_mfcc=40, max_pad_len=400) -> (1,40,400)
├── pipeline       : librosa.load(sr=16000) -> librosa.feature.mfcc(n_mfcc) ->
│                    truncate or zero-pad the time axis to 400 -> expand_dims(axis=0)
├── deps           : librosa, numpy, warnings
├── callers        : scripts_pc/01_process_audio.py
└── notes          : returns None on failure (silent skip); global warnings filter
```

### `utils/vision_ops.py` — 1,320 B — **status: CORE**
```
├── responsibility : raw image -> normalised cropped face tensor
├── key functions  : extract_face(image_path, target_size=(112,112)) -> (3,112,112)
├── pipeline       : cv2.imread (None on failure) -> BGR2GRAY -> Haar cascade
│                    detectMultiScale(1.1,4) -> largest box, else centre crop ->
│                    resize -> BGR2RGB -> transpose(2,0,1) -> (x-127.5)/128
├── deps           : cv2, numpy, os (os is unused)
├── callers        : scripts_pc/02_process_face.py
└── notes          : needs a RELATIVE path — cv2.imread cannot open non-ASCII
                     absolute paths on this host (measured)
```

---

## scripts_pc/ — offline pipeline (all are WRITE operations)

### `scripts_pc/00_setup_strangers.py` — 1,597 B — **status: ACTIVE**
```
├── responsibility : build the negative class for the current dataset layout
├── key functions  : setup()  (face_base="../data/raw/face", voice_base="../data/raw/voice")
├── behaviour      : fetch_olivetti_faces() first 5 identities × 10 images ->
│                    data/raw/face/Stranger_{0..4}/*.jpg (resized 112×112);
│                    np.random.uniform noise, 16 kHz, 2 s ->
│                    data/raw/voice/Stranger_{0..4}/*.wav
├── deps           : os, cv2, numpy, sklearn.datasets, soundfile
├── outputs        : 50 jpg + 25 wav  (overwrites without warning)
└── notes          : prints instructions telling the user to add their own data
```

### `scripts_pc/00_download_face_data.py` — 2,426 B — **status: LEGACY (dangerous)**
```
├── responsibility : earlier version of the face bootstrap (s1..s40 layout)
├── key functions  : prepare_face_data()
├── behaviour      : shutil.rmtree("../data/raw/face") then fetch_olivetti_faces()
│                    -> s1..s40/f"{i:03d}.jpg"
├── callers        : none (superseded by 00_setup_strangers.py)
└── status reason  : would DELETE data/raw/face/User_Me — the real user photos
```

### `scripts_pc/01_process_audio.py` — 1,660 B — **status: ACTIVE**
```
├── responsibility : batch MFCC extraction
├── key functions  : process_audio_data()
├── inputs         : cfg paths.raw_voice_dir, model.voice_n_mfcc, voice_max_pad_len
├── outputs        : data/processed/voice_features/<speaker>/<stem>.npy  (35 files)
├── deps           : os, glob, numpy, yaml, sys, utils.audio_ops
└── notes          : leftover branches for thchs30 / dev-clean / librispeech_mini
```

### `scripts_pc/02_process_face.py` — 1,421 B — **status: ACTIVE**
```
├── responsibility : batch face crop + normalise
├── key functions  : process_face_data()
├── outputs        : data/processed/face_features/<speaker>/<stem>.npy  (55 files)
├── deps           : os, glob, numpy, yaml, sys, utils.vision_ops
└── notes          : silently skips images whose extraction returns None
```


### `scripts_pc/03_train_single_models.py` — 5,816 B — **status: ACTIVE**
```
├── responsibility : train both extractors as 6-class classifiers
├── key classes    : FeatureDataset(data_dir) -> (tensor, int label) from folder names
├── key functions  : train_extractor(model, loader, num_classes, embed_dim, epochs, lr, save_path)
│                    main()
├── behaviour      : nn.Sequential(extractor, Linear(embed_dim, num_classes)) ·
│                    CrossEntropyLoss · Adam(lr from cfg) · epochs_single = 20 ·
│                    saves ONLY extractor.state_dict() (classifier head discarded)
├── deps           : os, glob, numpy, yaml, torch, torch.nn, torch.optim,
│                    torch.utils.data, sys, models.*
├── outputs        : weights/pytorch_pth/{voice,face}_extractor_weights.pth
└── notes          : no seed, no validation split, no checkpointing, no early stop
```

### `scripts_pc/04_train_fusion_scheme_b.py` — 3,992 B — **status: ACTIVE**
```
├── responsibility : train the attention fusion head ("方案B")
├── key functions  : train_fusion()
├── behaviour      : loads both extractors (eval, no_grad) · 200 hardcoded steps ·
│                    each step builds a 4-sample batch:
│                      2 × (User_Me voice, User_Me face)  label 1.0
│                      1 × (Stranger voice, User_Me face) label 0.0
│                      1 × (User_Me voice, Stranger face) label 0.0
│                    BCELoss + Adam(0.001) · np.random.choice without seed
├── deps           : torch, os, glob, numpy, yaml, sys, models.*
├── outputs        : weights/pytorch_pth/attention_fusion_weights.pth
└── notes          : no validation metric; cfg.train.epochs_fusion is ignored; the
                     mixed-modality negatives teach "modality consistency", which is
                     a different objective from identity verification
```

### `scripts_pc/05_export_and_quantize.py` — 4,582 B — **status: ACTIVE**
```
├── responsibility : PyTorch -> ONNX fp32 -> INT8 dynamic quantisation
├── key functions  : export_and_quantize_safely(model, dummy_input, model_name, out_dir,
│                    input_names, dynamic_axes);  main()
├── behaviour      : torch.onnx.export(opset 11, dynamic batch, constant folding), then
│                    shape_inference.quant_pre_process + quantize_dynamic(QInt8) via a
│                    temp dir "<drive>:\ort_temp_pi_quant" (long-path workaround)
├── deps           : torch, onnxruntime.quantization, os, shutil, yaml, sys, models.*
├── outputs        : weights/onnx_int8/{voice,face}_extractor{_fp32,_quant}.onnx and
│                    attention_fusion{_fp32,_quant}.onnx
└── notes          : quantize_dynamic on conv weights emits ConvInteger, which needs a
                     recent ONNX Runtime (see REPRODUCTION_STATUS.md)
```

---

## Files that are NOT part of the project (Layer 4 / do not publish)

| file | what it actually is | status |
| --- | --- | --- |
| `train_gan.py` | DCGAN homework (256×256, hymenoptera, matplotlib montage) | `UNKNOWN` (unrelated) |
| `1.py`, `2.py` | matplotlib homework (bar / pie / donut charts of exam scores and e-commerce sales) | `UNKNOWN` (unrelated) |
| `3.py` | downloads `hymenoptera_data.zip` from download.pytorch.org into `my_gan_data/` | `UNKNOWN` (unrelated) |
| `my_gan_data/` | 398 files, 45 MB — the extracted hymenoptera dataset | data (unrelated) |
| `pokemon.zip` | **not a zip** — an HTML page (`<!DOCTYP…`) renamed to `.zip` | junk |
| `live_capture.jpg` | 640×480 webcam capture, no EXIF | private candidate |
| `.idea/`, `__pycache__/` | IDE state + bytecode | ignore |
| `docs/` | **added by Phase 0** — not part of the original project | audit output |

## CatDog repository root (outside the project folder)

| path | content | status |
| --- | --- | --- |
| `data/cifar-10-python.tar.gz` + `data/cifar-10-batches-py/` | 340 MB CIFAR-10 | unrelated dataset |
| `.venv/` | Python 3.12.4 venv — torch 2.11.0, torchvision, ultralytics, yaml, soundfile; **no numpy, no pip module** | broken / ignore |
| `.venv1/` | only `pip` | empty / ignore |
| `.idea/` | `CatDog.iml`, `misc.xml` (Python 3.13), `workspace.xml` (2026-03-24) | ignore |
| `.vs/CatDog/v16/.suo` | Visual Studio binary state | ignore |
| `_phase0_audit/` | **added by Phase 0** — scripts, inventory, evidence JSON | audit output |
| `_workers_init.txt` | 0 content relevance, present in an unrelated workspace, not in CatDog | n/a |


# PROJECT_CONTEXT.md — CatDog / Pi_Multimodal_Auth_Project

Phase 0 evidence-based project cognition file.
Generated: 2026-09-22 · Status: **Phase 0 complete, no refactoring performed**

> ⚠️ An accidental re-training of 3 `.pth` files happened during this audit and was
> remediated. Read `PHASE0_REMEDIATION_NOTICE.md` before touching anything.

Evidence legend used everywhere in the Phase 0 documents:

| Tag | Meaning |
| --- | --- |
| `VERIFIED` | reproduced on this machine during Phase 0; command + output recorded |
| `PARTIALLY VERIFIED` | code/config evidence exists, behaviour only partly reproducible |
| `HISTORICAL` | provable only from artefacts/timestamps/logs, not re-runnable |
| `UNVERIFIED` | claimed or assumed somewhere, no evidence found |

---

## 1. 项目简介

`Pi_Multimodal_Auth_Project` 是一个 **人脸 + 声纹 双模态身份认证原型系统**。

* 两种模态各自用一个极小的自建 CNN 提取 **128 维 embedding**；
* 两个 embedding 经一个 **注意力门控融合网络**（`ModalAttentionFusion`）得到一个
  0~1 的"同一个人"置信度；
* 置信度与阈值 **0.85** 比较，输出 **认证通过 / 拒绝访问**；
* 训练侧在 PC 上完成，导出 **ONNX（fp32 + INT8 动态量化）**，
  推理侧代码位于 `app_pi/`，面向 **树莓派 (Raspberry Pi)** 边缘部署。

代码总量极小：**18 个 `.py` 文件 / 48 KB**（其中真正属于本项目的只有 13 个）。
没有 README、没有任何 `.md`、没有 Git 仓库、没有测试。

## 2. 项目真实目标（依据代码而非名称推断）

| 目标 | 代码证据 | 状态 |
| --- | --- | --- |
| 双模态（人脸+声纹）身份验证 | `app_pi/pi_main.py` 第 47-62 行；`models/attention_fusion.py` | `VERIFIED` |
| 融合判决使用单一置信度阈值 | `config/settings.yaml: inference.fusion_threshold: 0.85`；`pi_main.py:59-62` | `VERIFIED` |
| 面向树莓派的低算力设计 | `models/face_extractor.py:9` 注释"极简版 CNN，适应树莓派算力"；`requirements_pi.txt`；`app_pi/` | `PARTIALLY VERIFIED` |
| INT8 量化以加速边缘推理 | `scripts_pc/05_export_and_quantize.py:38 quantize_dynamic(QInt8)`；6 个 `.onnx` | `VERIFIED`（导出）／**加速效果被实测否定**（见 §16） |
| 在树莓派上实测性能并输出报告 | `pi_main.py:63-67` 打印"树莓派性能实测报告" | **`UNVERIFIED`** — 输入是随机数，且仓库内无任何 Pi 实测数据 |

**不是**这个项目的一部分（但同目录存在）：DCGAN 作业（`train_gan.py`,
`my_gan_data/`, `pokemon.zip`）、matplotlib 作业（`1.py`, `2.py`）、
PyTorch 教程数据集下载脚本（`3.py`）。

## 3. 当前技术栈（按实际执行的解释器取证）

原始解释器 **`D:\Python313\python.exe`（Python 3.13.1）** —— 依据：

* `.idea/misc.xml` → `project-jdk-name="Python 3.13"`；
* 项目内 `__pycache__/*.cpython-313.pyc`；
* `D:\Python313` 中同时存在 torch / onnx / onnxruntime / cv2 / librosa / sklearn。

| 库 | 原始环境 `D:\Python313` 实测 | 项目 `.venv` | anaconda base | Phase 0 隔离环境 |
| --- | --- | --- | --- | --- |
| Python | 3.13.1 | 3.12.4 | 3.12.4 | 3.12.x |
| torch | **2.7.1+cu118** | 2.11.0 | 2.6.0+cpu | — |
| torchvision | 有 | 0.26.0 | 0.21.0+cpu | — |
| onnx | 1.20.1 | ✗ | ✗ | 1.23.0 |
| onnxruntime | **1.24.1** | ✗ | ✗ | 1.17.3 / 1.20.1 |
| opencv | 4.13.0 | ✗ | 4.11.0 | 5.0.0 (headless) |
| librosa | 0.11.0 | ✗ | ✗ | 1.0.0 |
| scikit-learn | 1.8.0 | ✗ | 有 | — |
| numpy | 2.3.5 | **✗ 缺失** | 1.26.4 | 2.5.3 / 1.26.4 |
| soundfile / psutil / PyYAML | 有 | soundfile, yaml | psutil, yaml | 有 |
| sounddevice | **✗ 缺失** | ✗ | ✗ | 有 |

* `requirements_pc.txt`：`torch torchvision torchaudio librosa soundfile numpy opencv-python PyYAML tqdm onnx onnxruntime scikit-learn`（**全部未锁版本**；`torchaudio` 与 `tqdm` 在代码中从未被 import）。
* `requirements_pi.txt`：`onnxruntime==1.16.0`、`opencv-python-headless==4.8.1.78`、`sounddevice==0.4.6`、`psutil==5.9.5`、`numpy==1.24.3`、`PyYAML==6.0`。
* `.venv` 已损坏（无 numpy、无 pip 模块），`.venv1` 只有 pip —— **两者都不是可用环境**。

## 4. 目录结构

```
CatDog/
├── data/                        # 340 MB — CIFAR-10，与本项目无关
├── .venv/ .venv1/ .idea/ .vs/   # 环境与 IDE 残留
├── _phase0_audit/               # Phase 0 审计产物（本次生成，可整体删除）
└── Pi_Multimodal_Auth_Project/  # ← 真正的项目（619 文件 / 435 MB，含噪声）
    ├── app_pi/       pi_main.py, hardware_monitor.py      ← 边缘入口
    ├── config/       settings.yaml                        ← 唯一配置
    ├── models/       voice_extractor.py, face_extractor.py,
    │                 attention_fusion.py                  ← 核心模型
    ├── utils/        audio_ops.py, vision_ops.py          ← 预处理
    ├── scripts_pc/   00..05 (7 个脚本)                    ← 离线流水线
    ├── weights/onnx_int8/   6 个 .onnx (fp32+INT8)        ← 边缘权重
    ├── weights/pytorch_pth/ 3 个 .pth                     ← 训练权重
    ├── data/raw|processed/  人脸 / 声纹 原始与特征         ← 含隐私数据
    ├── my_gan_data/  398 文件 hymenoptera（GAN 作业，无关）
    ├── 1.py 2.py 3.py train_gan.py pokemon.zip live_capture.jpg（无关/垃圾）
    └── requirements_pc.txt / requirements_pi.txt
```

完整清单：`_phase0_audit/phase0_tree.txt`、`_phase0_audit/phase0_inventory.csv`
（619 文件 × sha256/size/mtime/类型/行数/编码）。

## 5. 程序入口

**唯一真正的端到端入口（边缘推理）：**

```bash
cd Pi_Multimodal_Auth_Project/app_pi
python pi_main.py          # 必须以此 cwd 运行（配置与权重路径都是 "../..."）
```

**离线流水线入口（PC，按序号顺序执行）：**

```bash
cd Pi_Multimodal_Auth_Project/scripts_pc
python 00_setup_strangers.py      # 造陌生人数据（Olivetti + 合成噪声）
python 01_process_audio.py        # 声纹特征提取
python 02_process_face.py         # 人脸特征提取
python 03_train_single_models.py  # 训练两个 extractor
python 04_train_fusion_scheme_b.py# 训练融合网络
python 05_export_and_quantize.py  # 导出 ONNX + INT8 量化
```

* 两处入口**都依赖 cwd**，脚本内部用 `open("../config/settings.yaml")`
  和 `glob("../data/...")`；`pi_main.py` 还使用 `from hardware_monitor import ...`
  这种同目录隐式导入。
* 没有 `argparse`、没有 CLI 参数、没有 `main.py`/`app.py`/`setup.py`/
  `pyproject.toml`、没有 `if __name__` 之外的调度层。
* 这些脚本是**写操作**（覆盖 `.npy` / `.pth` / `.onnx`），不能当作只读工具重复执行。

## 6. 完整调用链（逐函数取证）

### 6.1 边缘推理（`app_pi/pi_main.py`）

```
pi_main.py :: __main__                      (L70-76)
└── PiMultimodalAuth(config_path="../config/settings.yaml")     (L12-27)
    ├── yaml.safe_load(open(config_path))                       (L14-15)
    ├── ort.SessionOptions(); opts.intra_op_num_threads = 2     (L18-19)
    ├── ort.InferenceSession(<onnx>/voice_extractor_quant.onnx, opts)   (L23)
    ├── ort.InferenceSession(<onnx>/face_extractor_quant.onnx , opts)   (L24)
    ├── ort.InferenceSession(<onnx>/attention_fusion_quant.onnx, opts)  (L25)
    └── PiHardwareMonitor()                                     (L27)
        └── _check_if_pi() → 读 /sys/firmware/devicetree/base/model
└── authenticate()                                             (L38-67)
    ├── dummy_capture()                                        (L29-36)
    │      in : cfg['model']['voice_n_mfcc']=40,
    │           cfg['model']['voice_max_pad_len']=400,
    │           cfg['model']['face_image_size']=112
    │      out: voice (1,1,40,400) float32, face (1,3,112,112) float32
    │      ★ 由 np.random.randn 生成 —— 没有任何摄像头/麦克风采集代码
    ├── voice_sess.run(['output'], {'voice_input': …}) → (1,128)   (L47)
    ├── face_sess.run(['output'],  {'face_input' : …}) → (1,128)   (L48)
    ├── fusion_sess.run(['output'], {'voice_input':v_emb,
    │                                'face_input' :f_emb}) → (1,1) (L51-52)
    ├── score = …[0][0][0];  threshold = cfg['inference']['fusion_threshold'] (L52,59)
    ├── 输出 "认证通过 / 拒绝访问"                                  (L62)
    └── PiHardwareMonitor.get_system_stats() → 打印"树莓派性能实测报告" (L57,63-67)
```

### 6.2 离线流水线

```
00_setup_strangers.py :: setup()
├── sklearn.datasets.fetch_olivetti_faces()  → 前 5 人 × 10 张
│   └── cv2.resize(112×112) → cv2.imwrite  data/raw/face/Stranger_{0..4}/*.jpg
└── np.random.uniform(-0.1,0.1,16000*2) → soundfile.write 16 kHz
    → data/raw/voice/Stranger_{0..4}/*.wav      （合成噪声，不是真人语音）
00_download_face_data.py :: prepare_face_data()   （旧版，会 rmtree 整个 data/raw/face）
01_process_audio.py :: process_audio_data()
├── glob(data/raw/voice/**/*.{wav,flac})
├── speaker_id = basename(dirname(dirname(file)))  # 保留 4 个历史数据集名兜底分支
├── utils.audio_ops.extract_mfcc(path, 40, 400)
│   └── librosa.load(sr=16000) → librosa.feature.mfcc(n_mfcc=40)
│       → 截断/零填充到 400 帧 → expand_dims(0) → (1,40,400)
└── np.save(data/processed/voice_features/<speaker>/<stem>.npy)
02_process_face.py :: process_face_data()
├── glob(data/raw/face/**/*.{jpg,png})
├── utils.vision_ops.extract_face(path, (112,112))
│   ├── cv2.imread → 失败返回 None
│   ├── cv2.cvtColor(BGR→GRAY)
│   ├── cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
│   │   └── detectMultiScale(gray,1.1,4) → 取面积最大；无检测则中心裁剪
│   ├── resize(112,112) → BGR→RGB → transpose(2,0,1)
│   └── (x-127.5)/128 → (3,112,112) float32，范围约 [-1,1]
└── np.save(data/processed/face_features/<speaker>/<stem>.npy)
03_train_single_models.py :: main()
├── FeatureDataset(processed_voice_dir) → 目录名排序后作 label（6 类）
├── train_extractor(VoiceExtractor(40,128), num_classes=6, epochs=20, lr=1e-3)
│   └── nn.Sequential(extractor, nn.Linear(128, num_classes))
│       → CrossEntropyLoss + Adam → 只保存 extractor.state_dict()
├── 同法训练 FaceExtractor
└── torch.save → weights/pytorch_pth/{voice,face}_extractor_weights.pth
04_train_fusion_scheme_b.py :: train_fusion()
├── load 两个 extractor（eval + no_grad）
├── 数据：processed/*/User_Me/*.npy = 正样本；其余目录 = 陌生人
├── for epoch in range(200):        # ← 硬编码；config 中的 epochs_fusion=15 从未被读取
│   ├── 2×(User_Me voice + User_Me face)  label=1.0
│   ├── 1×(Stranger voice + User_Me face) label=0.0
│   ├── 1×(User_Me voice + Stranger face) label=0.0
│   └── ModalAttentionFusion.forward → BCELoss → Adam(1e-3) → step()
└── torch.save → weights/pytorch_pth/attention_fusion_weights.pth
05_export_and_quantize.py :: main()
├── torch.onnx.export(opset_version=11, do_constant_folding=True,
│                     dynamic_axes={0:'batch_size'}) → *_fp32.onnx
└── quant_pre_process + quantize_dynamic(weight_type=QInt8) → *_quant.onnx
    （临时目录 %DRIVE%\ort_temp_pi_quant，用于规避路径/权限问题）
```



## 7. 数据流

```
raw/face/User_Me/*.jpg    (5 张真实照片 960×1280)
raw/face/Stranger_0..4/   (50 张 Olivetti 112×112)
        │  cv2 Haar 检测 + 中心裁剪 + resize + (x-127.5)/128
        ▼
processed/face_features/<spk>/*.npy   (3,112,112) float32   ×55

raw/voice/User_Me/*.wav   (10 段真实录音 48 kHz, 9–14 s)
raw/voice/Stranger_0..4/  (25 段合成白噪声 16 kHz, 2 s)
        │  librosa 16 kHz 重采样 + MFCC(40) + pad/trunc 到 400 帧
        ▼
processed/voice_features/<spk>/*.npy  (1,40,400) float32 ×35
        │
        └────────────► 训练 / 推理 ◄────────────┘
                            │
                 128-d voice emb + 128-d face emb
                            │
                   ModalAttentionFusion → score ∈ (0,1)
                            │
                   score ≥ 0.85 ? 通过 : 拒绝
```

规模事实：正样本 **10 语音 × 5 人脸**，负样本 **25 合成噪声语音 × 50 公开数据集人脸**，
共 90 个特征文件 / 10.5 MB。**这是玩具规模的冒烟测试集，不是生物特征基准**。

## 8. 多模态架构（真正的融合点）

模态只有两个：**① 人脸图像（Face）② 语音 MFCC（Voice）**。
没有深度 / 红外 / 步态 / 手势等第三个模态。

```
模态 A：人脸
  raw jpg ─cv2 Haar─▶ 112×112 RGB ─(x-127.5)/128─▶ FaceExtractor(423,236 params) ─▶ 128-d
模态 B：语音
  raw wav ─librosa MFCC─▶ (1,40,400) ─▶ VoiceExtractor(109,184 params) ─▶ 128-d
                            │
        ❗融合点：models/attention_fusion.py::ModalAttentionFusion.forward (L31-44)
                            │
  v_feat = ReLU(v_proj(v_emb))                 # 128 → 64
  f_feat = ReLU(f_proj(f_emb))                 # 128 → 64
  w      = Softmax(Linear_2(ReLU(Linear_1([v_feat; f_feat]))) )   # [B,2]，和为 1
  fused  = v_feat * w[:,0] + f_feat * w[:,1]   # ← 加权求和，不是拼接
  score  = Sigmoid(Linear_4(ReLU(BN(Linear_0(fused)))))          # [B,1] ∈ (0,1)
```

融合范式判定（依据代码，不依据描述）：

| 融合范式 | 是否 | 依据 |
| --- | --- | --- |
| Early / 像素级融合 | ✗ | 原始图像与波形从未拼接 |
| Late / embedding 级融合 | **✔** | 融合发生在两个 128-d embedding 之后 |
| 特征拼接 | ✗ | `torch.cat` 仅用于计算门控权重，特征本身是加权求和 |
| 分数级融合（加权打分） | ✗ | 两个分支没有各自的分数，只有 embedding |
| **Learned attention-gated feature fusion** | **✔** | 2 路 softmax 软门控 + MLP 二分类头 |
| Decision fusion | 部分 | 最终判决确实是对融合分数做单阈值二值化 |

## 9. 模型清单（参数为脚本实测值，非估算）

由 `state_dict()` 直接统计（`D:\Python313`，torch 2.7.1）：

| Model | Framework | Input | Output | Parameters | state_dict bytes | 原 .pth 大小 | 被谁使用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VoiceExtractor` | PyTorch / ONNX opset 11 | `voice_input` (B,1,40,400) f32 | `output` (B,128) | **109,184** | 436,736 | 440,749 | `03`, `05` |
| `FaceExtractor` | PyTorch / ONNX | `face_input` (B,3,112,112) f32 | `output` (B,128) | **423,236** | 1,692,960 | 1,703,845 | `03`, `05` |
| `ModalAttentionFusion` | PyTorch / ONNX | `voice_input` (B,128), `face_input` (B,128) | `output` (B,1) | **27,140** | 108,564 | 115,319 | `04`, `05` |
| 合计 | | | | **559,560** | 2,238,260 | | |

ONNX 文件与**本次实测的可加载性**（ORT 版本极度敏感，见 §16）：

| file | size | ORT 1.17.3 | ORT 1.20.1 | ORT 1.24.1 | sha256 前16位 |
| --- | --- | --- | --- | --- | --- |
| `voice_extractor_fp32.onnx` | 438,936 | ✔ | ✔ | ✔ | 191697363BE7EB0A |
| `voice_extractor_quant.onnx` | 117,288 | ✗ opset | ✗ ConvInteger | **✔** | EC25946393150E25 |
| `face_extractor_fp32.onnx` | 1,687,815 | ✔ | ✔ | ✔ | 3949BE0816C813A9 |
| `face_extractor_quant.onnx` | 432,865 | ✗ opset | ✗ ConvInteger | **✔** | 0BA342DDF0B5EE34 |
| `attention_fusion_fp32.onnx` | 112,106 | ✔ | ✔ | ✔ | D8272A3986BB7735 |
| `attention_fusion_quant.onnx` | 40,172 | ✗ opset | ✔ | ✔ | 34C2C0586A69018D |

实际用过的推理框架：**只有 PyTorch 与 ONNX Runtime**。
仓库内 **没有** NCNN / OpenVINO / TFLite / TensorRT / RKNN 的任何文件、代码或依赖
（`grep -i ncnn|openvino|tflite|cv2.dnn` 零命中）。

## 10. 推理流程（两条路径，均已实测）

| | PyTorch 路径 | ONNX 路径（边缘实际使用） |
| --- | --- | --- |
| 权重 | `weights/pytorch_pth/*.pth` | `weights/onnx_int8/*_quant.onnx` |
| 入口 | `03→04` 训练脚本内部；无独立推理脚本 | `app_pi/pi_main.py::authenticate` |
| 实测单样本延迟 | **2.02 ms** | fp32 **1.25–1.48 ms** / INT8 **33.4 ms** |
| 实测测试机 | 本机 x86 CPU，torch 2.7.1+cu118 | 本机 x86 CPU，ORT 1.24.1 |
| 加速效果 | — | **INT8 比 fp32 慢约 22 倍**（`ConvInteger` 无优化 CPU kernel） |

**没有任何树莓派实测数据存在于仓库中。** 上面所有延迟都是本机 x86 数字。

## 11. 融合机制

见 `CORE_ARCHITECTURE.md` §4。结论：**late / embedding-level / learned
attention-gated fusion**，双路 softmax 门控（权重和为 1）+ 加权求和 + MLP 二分类头。

## 12. 决策机制

单全局阈值 `score >= 0.85`（`config/settings.yaml`）→ `认证通过 / 拒绝访问`。
没有 per-user 校准、没有活体检测、没有质量门控、没有重试策略、没有限流。

## 13. 配置机制

`config/settings.yaml` 是唯一配置源，被 5 个脚本以 `open("../config/settings.yaml")`
各读一次（无共享 loader、无 schema 校验、无环境变量覆盖）。
**配置漂移已实测确认**：`train.epochs_fusion: 15` 在代码中**从未被引用**，
`04_train_fusion_scheme_b.py` 硬编码 `range(200)`。

## 14. 实验系统

**不存在独立实验框架。** 具体表现：

* 无 `experiments/`、`results/`、`logs/`、`benchmark/`、`output/` 目录（实测：零命中）；
* 无 accuracy / precision / recall / F1 / latency / FPS / 内存 / 混淆矩阵的落盘记录；
* 训练脚本只 `print` 每轮 Loss/Acc，输出不写文件、不带时间戳、不入库；
* `.idea/workspace.xml` 中含 PyCharm coverage 运行记录（6 个脚本、2026-04-19），
  这是**唯一**的执行痕迹证据；
* 唯一的"结果"是 `05` 导出的 6 个 `.onnx` 与 3 个 `.pth`。

Phase 0 自己测出的数字（可复现、有脚本）见 §16，与项目原有记录无关。

## 15. 测试系统

**不存在。** 无 `test/`、`tests/`、`pytest`/`unittest` 引用、无 CI 配置、
无 `.github/`、无 `conftest.py`、无 mock、无 assert（`train_gan.py` 亦无）。

## 16. 历史边缘部署（Historical Edge Deployment Audit）

搜索范围：`Raspberry / Pi / aarch64 / arm64 / ARM / NCNN / OpenVINO / ONNX Runtime /
TFLite / Edge / Deployment / cv2.dnn`，命中文件全部列出（`_phase0_audit/_edge_scan.txt`）：

| 命中的文件 | 命中内容 |
| --- | --- |
| `app_pi/hardware_monitor.py` L10 | `if 'Raspberry Pi' in m.read()` — 读设备树型号 |
| `app_pi/pi_main.py` L22,39,63 | 注释/打印"树莓派"、加载 `_quant.onnx` |
| `models/face_extractor.py` L9 | 注释"极简版 CNN，适应树莓派算力" |
| `config/settings.yaml` L8 | `onnx_weights_dir` |
| `requirements_pi.txt` | `onnxruntime==1.16.0` 等 6 项 |
| `scripts_pc/05_export_and_quantize.py` | `torch.onnx.export` + `quantize_dynamic` |

**零命中**：`aarch64`, `arm64`, `ncnn`, `openvino`, `tflite`, `cv2.dnn`,
`部署脚本`, `systemd`, `scp`, `ssh`, `rsync`, 任何 `.sh` 部署脚本。

| 分级 | 结论 |
| --- | --- |
| **LEVEL A（明确证明）** | PC 端**确实**完成了「训练→导出 ONNX fp32→INT8 动态量化」全流程。证据：6 个 `.onnx`（mtime 2026-04-19 14:59:28）+ 3 个 `.pth`（14:59:15/22）+ PyCharm coverage 运行记录（14:59:00–14:59:25）+ `requirements_pi.txt` + `app_pi/` 代码。 |
| **LEVEL B（有部署代码，无历史运行证据）** | 存在完整的边缘推理代码与依赖清单，且本机实测该代码**可在 ORT 1.24.1 上跑通全链路**；但**没有**任何 Pi 侧日志、截图、性能数据、部署脚本或 systemd 服务。 |
| **LEVEL C（仅有理论/依赖支持）** | 所有"树莓派性能"数字。`pi_main.py` 的输入是 `np.random.randn`，`hardware_monitor` 非 Pi 时温度恒为 0.0 —— 即使真有报告输出，也不能作为性能证据。 |

> Historical deployment evidence only. Current hardware reproduction unavailable.

## 17. 当前可复现能力（本机实测）

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| 人脸预处理逐位复现 | ✔ | `probe_preprocessing.json`: 55/55 `identical=True` |
| 声纹预处理逐位复现 | ✔ | 同上：35/35 `identical=True` |
| 边缘入口加载 INT8 模型并推理 | ✔（需 ORT ≥ 1.21/1.24） | `_pi_harness.txt`：三模型成功建 session，跑出 score |
| fp32 ONNX 端到端推理 | ✔ | 任意 ORT（1.17.3/1.20.1/1.24.1） |
| PyTorch 权重端到端推理（含判决） | ✔ | `inplace_verification.json` |
| 训练脚本可运行 | ✔（会覆写权重） | 03/04 在审计中被误执行并成功完成 |
| 输出 ≥0.85 / <0.05 的分离度 | ✔（玩具级数据） | genuine 0.9182 [0.8943,0.9293]；impostor ≤ 0.0483 |
| Pi 实机运行/性能 | ✘ | 无硬件、无历史数据 |
| 真实摄像头/麦克风采集 | ✘ | 代码不存在（`dummy_capture`） |
| 训练可复现（同 seed 同权重） | ✘ | 无任何随机种子 |
| Git 历史 | ✘ | 无 `.git` |


## 18. 当前不可复现能力

* 树莓派实机推理、温度/CPU/RAM 报告、任何 FPS 或延迟数字；
* 真实摄像头 + 麦克风采集（代码不存在）；
* 原始训练随机过程（无 seed，重跑得到不同权重）；
* INT8 是否真的"加速"（实测在本机 x86 上慢了 22 倍，Pi 上结果未知）；
* 03/04 在**原始解释器 `D:\Python313`** 下的原始权重（无 seed，无法回放）。

## 19. 已知问题（按严重程度）

1. **【部署阻断】** `requirements_pi.txt` 锁 `onnxruntime==1.16.0`，而随仓库交付的
   `voice/face_extractor_quant.onnx` 使用 `ConvInteger`，在 ORT ≤1.20.1 上
   抛 `NOT_IMPLEMENTED`；实测 `pi_main.py` 在这个组合下**初始化即失败**
   （`_run_pi_main.txt`）。
2. **【功能空缺】** 边缘入口无真实采集：`dummy_capture()` 返回随机数，
   注释明说"实际部署时需要换成你的 opencv 和 sounddevice 真实采集代码"。
3. **【崩溃 bug】** `pi_main.py` L62 打印 🟢/🔴，在 GBK 控制台抛
   `UnicodeEncodeError`，被外层 except 吞掉并误报为"初始化失败"。
4. **【危险】** `00_download_face_data.py` 会 `rmtree` 整个 `data/raw/face`，
   会删掉用户本人 5 张真实照片。
5. **【依赖不可安装】** `onnxruntime==1.16.0` 在 Python 3.13 上没有 wheel
   （实测镜像仅提供 ≥1.17.0），而项目原始解释器正是 3.13.1。
6. **【配置漂移】** `train.epochs_fusion` 从未被读取；`fusion_hidden_dim` 未被 04/05 使用。
7. **【不可复现】** 全流程无随机种子。
8. **【无文档/无测试/无 Git】** 0 个 `.md`、0 个测试、无版本控制。
9. **【PATH 敏感】** 所有脚本依赖 cwd；非 ASCII 绝对路径下 `cv2.imread` 直接失败。
10. **【环境损毁】** 项目自带 `.venv` 已不可用（缺 numpy、缺 pip 模块），
    `.venv1` 只有 pip；真正的环境在项目外的 `D:\Python313`。

## 20. 代码技术债

| 类别 | 具体 |
| --- | --- |
| 分层 | 完全扁平：脚本直接 import 模型与 utils，模型层不依赖业务层（唯一优点） |
| 耦合 | 配置读取、路径拼接、类别枚举硬编码在 5 个脚本中重复 5 遍 |
| 硬编码 | `range(200)`、`lr=0.001`、4 样本 batch 构造、`"User_Me"` 字符串、`0.85` |
| 全局状态 | 脚本级全局 `device`、`DATA_PATH`（`train_gan.py`） |
| 重复模型加载 | 每个脚本各自实例化并 load_state_dict |
| 命名 | `1.py / 2.py / 3.py`、`attention_fusion` 里 `v_proj/f_proj` 与注释不一致 |
| 死代码 | `00_download_face_data.py`（被 `00_setup_strangers.py` 取代）；`01_process_audio.py` 中 4 个不存在数据集的兜底分支 |
| 未用依赖 | `torchaudio`、`tqdm`（requirements_pc 中列出，代码从未 import） |
| 日志 | 全部 `print`，无 logging、无文件名/时间戳，不可追溯 |
| 安全 | 无凭据泄露；但 `.idea/workspace.xml` 含真实账户绝对路径 |

## 21. 开源风险（摘要，详见 `OPEN_SOURCE_RISK.md`）

* `data/raw/face/User_Me/*.jpg`（5 张真人照片）、`data/raw/voice/User_Me/*.wav`
  （10 段真人录音）、`data/processed/**/User_Me/*.npy`（生物特征模板）、
  `live_capture.jpg` → **PRIVATE，禁止上传**。
* `data/raw/face/Stranger_*`（Olivetti 派生）→ 可重新生成，建议不直接提交。
* 无 LICENSE；无第三方模型/数据集的许可声明（Olivetti 属学术用途）。
* 大二进制：`my_gan_data` 45 MB、`data/cifar-10` 170 MB+340 MB（与本项目无关）。
* `.idea/.vs/.venv/.venv1` 与 `pokemon.zip`（HTML 错误页冒充 zip）→ REMOVE。

## 22. 可开源文件（Layer 1 + Layer 2）

```
app_pi/hardware_monitor.py, app_pi/pi_main.py        (需先修 F2/F3)
config/settings.yaml
models/{voice_extractor,face_extractor,attention_fusion}.py
utils/{audio_ops,vision_ops}.py
scripts_pc/00_setup_strangers.py, 01..05              (00_download_face_data.py 建议删除)
weights/onnx_int8/*.onnx, weights/pytorch_pth/*.pth   (~5 MB, 需确认许可)
requirements_pc.txt / requirements_pi.txt             (需修正版本)
新增: README.md, LICENSE, docs/*, .gitignore, sample/synthetic data generator
```

## 23. 不可开源文件（Layer 4）

```
data/raw/face/User_Me/           5 张真人照片
data/raw/voice/User_Me/          10 段真人录音（含文件名中的日期）
data/processed/face_features/User_Me/, voice_features/User_Me/   生物特征模板
live_capture.jpg                 摄像头抓拍
data/raw/face/Stranger_*         可重新生成，但属第三方数据集派生
.idea/workspace.xml              含真实 Windows 账户绝对路径
.venv/, .venv1/, .vs/            环境残留
my_gan_data/, pokemon.zip, 1.py, 2.py, 3.py, train_gan.py   无关内容
CatDog/data/cifar-10*            无关的 170 MB+ 数据集
```

## 24. 推荐重构方向（Phase 1 之后再动手，此处仅记录）

1. **先修部署阻断**：统一 `requirements_pi.txt` 与量化模型的 ORT 版本；
   或改用 `QLinearConv`（QDQ）量化路径，使 INT8 在 CPU/ARM 上真正可用。
2. **补齐采集层**：实现 `capture.py`（cv2 摄像头 + sounddevice），
   与 `utils/` 中的预处理严格共用同一函数，消除 train/serve skew。
3. **建立单一入口**：`run_pc.py` / `run_pi.py` + `argparse`，路径基于 `__file__`。
4. **引入配置加载器**：`config.py` 一次性读取 + dataclass 校验，删除 5 处重复。
5. **加评测脚本**：`evaluate.py` 输出阈值扫描、FAR/FRR/EER（明确标注数据规模与
   合成噪声的限制），把 Phase 0 的 `inplace_verification.json` 变成可重跑基线。
6. **加种子与可复现性**：`set_seed()`、`torch.use_deterministic_algorithms`。
7. **拆分项目边界**：把 GAN/绘图作业移出该仓库。
8. **补工程基础**：README、LICENSE、`.gitignore`、`tests/`、`Makefile`/`noxfile`。

## 25. 推荐 GitHub 项目定位

**定位**：*"A minimal, readable two-modality (face + voice) identity-verification
prototype with an attention-gated fusion head, end-to-end reproducible from raw
data to INT8 ONNX, targeting Raspberry-Pi-class edge devices."*

可以自豪地写的（全部 `VERIFIED`）：

* 从零手写双模态特征提取 + 注意力门控融合网络（约 56 万参数，权重 < 5 MB）；
* 完整可复现的 6 步数据/训练/导出流水线，预处理**逐位可复现**；
* ONNX 导出 + INT8 动态量化，并附带**量化前后精度对比**；
* 在真实数据上 genuine/impostor 分数完全分离（0.9182 vs ≤0.0483）。

**必须诚实标注的**：

* 数据集是玩具规模，负样本部分为合成噪声；**不构成生物特征基准或安全结论**；
* 边缘侧无真实采集、无 Pi 实测数据，只有 *historical deployment evidence*；
* INT8 在本机 x86 上**没有**加速（33.4 ms vs 1.48 ms），因此不要把"量化加速"
  写成已获得的收益；
* 量化模型依赖较新的 onnxruntime。


# ============================================================================
# FROZEN (Phase 0 audit) - historical PC pipeline. Superseded by scripts/.
# Kept in the repository as provenance for how the shipped weights were
# produced. Do not re-run it: it targeted the pre-Phase-1 layout
# (models/, utils/, weights/pytorch_pth) and its training step wrote
# straight back into weights/, which is exactly the Phase 0 incident.
# The three modern replacements are:
#   scripts/data/01_build_voice_features.py  (was 01_process_audio.py)
#   scripts/data/02_build_face_features.py   (was 02_process_face.py)
#   scripts/train/train_extractors.py + train_fusion.py
#   scripts/export.py                        (was 05_export_and_quantize.py)
# Set MULTIMODAL_AUTH_RUN_LEGACY=1 to bypass this guard at your own risk.
# ============================================================================
import os as _os, sys as _sys

if _os.environ.get("MULTIMODAL_AUTH_RUN_LEGACY") != "1":
    _sys.exit(
        "FROZEN legacy script (scripts_pc/): nothing was executed.\n"
        "Its imports (models.*, utils.*) and weight paths no longer exist."
    )

import os
import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import soundfile as sf


def setup():
    # 1. 准备陌生人的人脸 (使用 sklearn)
    face_base = "../data/raw/face"
    print("[*] 正在下载陌生人脸部数据...")
    faces = fetch_olivetti_faces()
    # 我们只取前 5 个陌生人，每个人的 10 张图
    for i in range(5):
        stranger_dir = os.path.join(face_base, f"Stranger_{i}")
        os.makedirs(stranger_dir, exist_ok=True)
        for j in range(10):
            img = (faces.images[i * 10 + j] * 255).astype(np.uint8)
            img = cv2.resize(img, (112, 112))
            cv2.imwrite(os.path.join(stranger_dir, f"{j}.jpg"), img)

    # 2. 准备模拟的陌生人语音 (防止你没有别人的语音)
    voice_base = "../data/raw/voice"
    print("[*] 正在生成模拟的陌生人语音...")
    for i in range(5):
        stranger_v_dir = os.path.join(voice_base, f"Stranger_{i}")
        os.makedirs(stranger_v_dir, exist_ok=True)
        for j in range(5):
            # 生成随机噪声模拟不同人的声音
            dummy_audio = np.random.uniform(-0.1, 0.1, 16000 * 2)
            sf.write(os.path.join(stranger_v_dir, f"{j}.wav"), dummy_audio, 16000)

    print("=" * 50)
    print("✅ 陌生人数据准备完毕！")
    print("👉 现在请把【你自己的 5 张照片】放入 data/raw/face/User_Me")
    print("👉 请把【你自己的 10 个音频】放入 data/raw/voice/User_Me")
    print("=" * 50)


if __name__ == "__main__":
    setup()
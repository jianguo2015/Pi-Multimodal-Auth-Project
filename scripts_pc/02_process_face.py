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
import glob
import numpy as np
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vision_ops import extract_face


def process_face_data():
    with open("../config/settings.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg['paths']['raw_face_dir']
    out_dir = cfg['paths']['processed_face_dir']
    img_size = (cfg['model']['face_image_size'], cfg['model']['face_image_size'])

    os.makedirs(out_dir, exist_ok=True)
    images = glob.glob(os.path.join(raw_dir, '**', '*.jpg'), recursive=True) + \
             glob.glob(os.path.join(raw_dir, '**', '*.png'), recursive=True)

    print(f"[*] 开始处理 {len(images)} 张人脸图像...")

    for i, file_path in enumerate(images):
        speaker_id = os.path.basename(os.path.dirname(file_path))
        feat = extract_face(file_path, target_size=img_size)

        if feat is not None:
            save_dir = os.path.join(out_dir, speaker_id)
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.basename(file_path).split('.')[0]
            np.save(os.path.join(save_dir, f"{file_name}.npy"), feat)

        if (i + 1) % 50 == 0:
            print(f" -> 进度: {i + 1}/{len(images)}")

    print("[+] 人脸特征提取完成！")


if __name__ == "__main__":
    process_face_data()
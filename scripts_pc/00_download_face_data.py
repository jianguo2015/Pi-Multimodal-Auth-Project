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
import shutil
from sklearn.datasets import fetch_olivetti_faces


def prepare_face_data():
    base_dir = "../data/raw/face"

    # 清理旧目录
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    print("=" * 65)
    print("[*] 正在通过 scikit-learn 获取 Olivetti/ORL 人脸数据集...")
    print("[*] 这会自动连接稳定的学术镜像源，下载量约 4MB...")
    print("=" * 65)

    try:
        # 1. 抓取数据 (如果本地没有会自动下载)
        # 这个接口比 GitHub 链接稳定得多
        faces_data = fetch_olivetti_faces(shuffle=False, random_state=42)

        # faces_data.images 是 (400, 64, 64) 的浮点数组 [0, 1]
        # faces_data.target 是 (400,) 的标签 [0...39]

        images = faces_data.images
        labels = faces_data.target

        print(f"[+] 成功获取数据：共 {len(images)} 张图片，涉及 40 个不同的人物。")
        print("[*] 正在转换数据格式并存储到磁盘...")

        # 2. 将数据保存为项目所需的目录结构
        for i in range(len(images)):
            person_id = labels[i]
            person_dir = os.path.join(base_dir, f"s{person_id + 1}")  # s1 到 s40

            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            # 转换浮点数 [0, 1] 到 uint8 [0, 255]
            img_uint8 = (images[i] * 255).astype(np.uint8)

            # 因为是 64x64，我们稍微放大一点，方便后续的人脸检测流程
            img_resized = cv2.resize(img_uint8, (112, 112))

            save_path = os.path.join(person_dir, f"{i:03d}.jpg")
            cv2.imwrite(save_path, img_resized)

        print("=" * 65)
        print(f"✅ 完美收工！")
        print(f"👉 数据已整理至: {os.path.abspath(base_dir)}")
        print(f"👉 请依次运行: 02_process_face.py -> 03_train_single_models.py")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n如果这里也报错，说明你的网络完全无法连接外部学术源。")
        print("建议：")
        print("1. 确保已安装 sklearn: pip install scikit-learn")
        print("2. 检查网络代理设置。")


if __name__ == "__main__":
    prepare_face_data()
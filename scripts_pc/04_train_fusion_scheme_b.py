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

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import numpy as np
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.voice_extractor import VoiceExtractor
from models.face_extractor import FaceExtractor
from models.attention_fusion import ModalAttentionFusion


def train_fusion():
    with open("../config/settings.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    weights_dir = cfg['paths']['pytorch_weights_dir']
    embed_dim = cfg['model']['embed_dim']

    # 1. 加载提取器
    voice_net = VoiceExtractor(n_mfcc=cfg['model']['voice_n_mfcc'], embed_dim=embed_dim)
    face_net = FaceExtractor(embed_dim=embed_dim)
    voice_net.load_state_dict(torch.load(os.path.join(weights_dir, "voice_extractor_weights.pth")))
    face_net.load_state_dict(torch.load(os.path.join(weights_dir, "face_extractor_weights.pth")))
    voice_net.eval()
    face_net.eval()

    # 2. 分类数据路径
    # 本人数据
    me_v_files = glob.glob(os.path.join(cfg['paths']['processed_voice_dir'], "User_Me", "*.npy"))
    me_f_files = glob.glob(os.path.join(cfg['paths']['processed_face_dir'], "User_Me", "*.npy"))
    # 陌生人数据 (排除 User_Me 文件夹)
    stranger_v_dirs = [d for d in glob.glob(os.path.join(cfg['paths']['processed_voice_dir'], "*")) if
                       "User_Me" not in d]
    stranger_f_dirs = [d for d in glob.glob(os.path.join(cfg['paths']['processed_face_dir'], "*")) if
                       "User_Me" not in d]

    st_v_files = []
    for d in stranger_v_dirs: st_v_files.extend(glob.glob(os.path.join(d, "*.npy")))
    st_f_files = []
    for d in stranger_f_dirs: st_f_files.extend(glob.glob(os.path.join(d, "*.npy")))

    print(
        f"[*] 样本统计 -> 本人音频:{len(me_v_files)} 本人脸:{len(me_f_files)} | 陌生人音频:{len(st_v_files)} 陌生人脸:{len(st_f_files)}")

    # 3. 训练融合模型
    fusion_model = ModalAttentionFusion(voice_dim=embed_dim, face_dim=embed_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

    fusion_model.train()

    for epoch in range(200):  # 增加迭代次数
        optimizer.zero_grad()

        # --- 构造一个 Mini-Batch (2个正样本, 2个负样本) ---
        v_list, f_list, labels = [], [], []

        # 正样本 (本人+本人)
        for _ in range(2):
            v_list.append(torch.tensor(np.load(np.random.choice(me_v_files))).float())
            f_list.append(torch.tensor(np.load(np.random.choice(me_f_files))).float())
            labels.append([1.0])

        # 负样本 (本人脸+陌生人声音)
        for _ in range(1):
            v_list.append(torch.tensor(np.load(np.random.choice(st_v_files))).float())
            f_list.append(torch.tensor(np.load(np.random.choice(me_f_files))).float())
            labels.append([0.0])

        # 负样本 (陌生人脸+本人声音)
        for _ in range(1):
            v_list.append(torch.tensor(np.load(np.random.choice(me_v_files))).float())
            f_list.append(torch.tensor(np.load(np.random.choice(st_f_files))).float())
            labels.append([0.0])

        v_raw = torch.stack(v_list)
        f_raw = torch.stack(f_list)
        target = torch.tensor(labels, dtype=torch.float32)

        with torch.no_grad():
            v_emb = voice_net(v_raw)
            f_emb = face_net(f_raw)

        output = fusion_model(v_emb, f_emb)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/200], Loss: {loss.item():.6f}")

    torch.save(fusion_model.state_dict(), os.path.join(weights_dir, "attention_fusion_weights.pth"))
    print("[+] 融合模型强化训练完成！")


if __name__ == "__main__":
    train_fusion()
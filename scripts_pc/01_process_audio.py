import os
import glob
import numpy as np
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_ops import extract_mfcc


def process_audio_data():
    with open("../config/settings.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg['paths']['raw_voice_dir']
    out_dir = cfg['paths']['processed_voice_dir']
    n_mfcc = cfg['model']['voice_n_mfcc']
    max_len = cfg['model']['voice_max_pad_len']

    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(raw_dir, '**', '*.flac'), recursive=True) + \
            glob.glob(os.path.join(raw_dir, '**', '*.wav'), recursive=True)

    print(f"[*] 开始处理 {len(files)} 个音频文件...")

    for i, file_path in enumerate(files):
        # 根据目录结构提取 Speaker ID
        parent_dir = os.path.dirname(file_path)
        speaker_id = os.path.basename(os.path.dirname(parent_dir))
        if speaker_id in ['thchs30', 'dev-clean', 'librispeech_mini', 'voice']:
            speaker_id = os.path.basename(parent_dir)

        feat = extract_mfcc(file_path, n_mfcc, max_len)
        if feat is not None:
            save_dir = os.path.join(out_dir, speaker_id)
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.basename(file_path).split('.')[0]
            np.save(os.path.join(save_dir, f"{file_name}.npy"), feat)

        if (i + 1) % 100 == 0:
            print(f" -> 进度: {i + 1}/{len(files)}")

    print("[+] 语音特征提取完成！")


if __name__ == "__main__":
    process_audio_data()
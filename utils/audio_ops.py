import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')  # 忽略librosa的某些由于格式引起的警告


def extract_mfcc(file_path, n_mfcc=40, max_pad_len=400):
    """
    提取MFCC特征，并将时间维度统一裁剪或填充到 max_pad_len
    返回 shape: (1, n_mfcc, max_pad_len)
    """
    try:
        # librosa 默认支持 .flac, .wav
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # 长度对齐
        if mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # 增加 channel 维度以适配 CNN，如 (1, 40, 400)
        return np.expand_dims(mfcc, axis=0)
    except Exception as e:
        print(f"音频处理失败 {file_path}: {e}")
        return None
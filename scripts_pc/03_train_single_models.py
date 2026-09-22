import os
import glob
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

# 确保能找到其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.voice_extractor import VoiceExtractor
from models.face_extractor import FaceExtractor


# ==========================================
# 1. 定义通用的特征数据集加载器
# ==========================================
class FeatureDataset(Dataset):
    def __init__(self, data_dir):
        """加载提取好的 .npy 特征文件，并自动为文件夹(Speaker ID)打上数字标签"""
        self.file_paths = []
        self.labels = []
        self.label_map = {}

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"找不到数据目录: {data_dir}，请先运行 01/02 数据处理脚本！")

        speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        speaker_dirs.sort()  # 保证标签顺序一致

        for idx, spk in enumerate(speaker_dirs):
            self.label_map[spk] = idx
            spk_dir = os.path.join(data_dir, spk)
            files = glob.glob(os.path.join(spk_dir, "*.npy"))
            self.file_paths.extend(files)
            self.labels.extend([idx] * len(files))

        self.num_classes = len(self.label_map)
        print(f" -> 加载了 {len(self.file_paths)} 个样本，共 {self.num_classes} 个类别 (说话人)。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feat = np.load(self.file_paths[idx])
        label = self.labels[idx]
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ==========================================
# 2. 通用的单模态模型训练函数
# ==========================================
def train_extractor(extractor_model, dataloader, num_classes, embed_dim, epochs, lr, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> 使用设备: {device}")

    # 因为我们的提取器只输出 128维 Embedding，所以为了进行分类训练，
    # 我们需要在它后面临时外挂一个全连接分类层 (Classifier Head)。
    classifier_head = nn.Linear(embed_dim, num_classes)

    # 将特征提取器和分类头组合成完整的分类网络
    model = nn.Sequential(extractor_model, classifier_head).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch[{epoch + 1}/{epochs}] | Loss: {total_loss / len(dataloader):.4f} | Acc: {acc:.2f}%")

    # 【核心！】训练完成后，我们只要前面的特征提取器权重，把后面的分类头扔掉！
    # 这样提取器以后就只会吐出 128维 的高维特征，完美对接 04 步的融合网络
    torch.save(extractor_model.state_dict(), save_path)
    print(f"[+] 提取器权重已保存至: {save_path}\n")


# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    with open("../config/settings.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    weights_dir = cfg['paths']['pytorch_weights_dir']
    os.makedirs(weights_dir, exist_ok=True)

    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs_single']
    lr = cfg['train']['learning_rate']
    embed_dim = cfg['model']['embed_dim']

    # --- 阶段一：训练声纹提取器 ---
    print("\n" + "=" * 40)
    print("[*] 开始训练声纹模型 (voice Extractor)")
    print("=" * 40)
    voice_dataset = FeatureDataset(cfg['paths']['processed_voice_dir'])
    if len(voice_dataset) > 0:
        voice_loader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=True)
        voice_net = VoiceExtractor(n_mfcc=cfg['model']['voice_n_mfcc'], embed_dim=embed_dim)

        save_path = os.path.join(weights_dir, "voice_extractor_weights.pth")
        train_extractor(voice_net, voice_loader, voice_dataset.num_classes, embed_dim, epochs, lr, save_path)
    else:
        print("未找到声纹数据，跳过训练。")

    # --- 阶段二：训练人脸提取器 ---
    print("\n" + "=" * 40)
    print("[*] 开始训练视觉模型 (Face Extractor)")
    print("=" * 40)
    face_dataset = FeatureDataset(cfg['paths']['processed_face_dir'])
    if len(face_dataset) > 0:
        face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)
        face_net = FaceExtractor(embed_dim=embed_dim)

        save_path = os.path.join(weights_dir, "face_extractor_weights.pth")
        train_extractor(face_net, face_loader, face_dataset.num_classes, embed_dim, epochs, lr, save_path)
    else:
        print("未找到人脸数据，跳过训练。")

    print("🎉 单模态网络训练全部完成！你可以接着运行 04_train_fusion_scheme_b.py 了。")


if __name__ == "__main__":
    main()
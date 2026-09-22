import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceExtractor(nn.Module):
    def __init__(self, n_mfcc=40, embed_dim=128):
        super(VoiceExtractor, self).__init__()
        # 输入 shape: (batch_size, 1, n_mfcc, time_steps)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)

        # 使用自适应池化，无视时间维度的长度，强制输出固定的宽高 (1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平 (batch_size, 128)

        emb = self.fc(x)  # (batch_size, embed_dim)
        return emb
import torch
import torch.nn as nn


class ModalAttentionFusion(nn.Module):
    def __init__(self, voice_dim=128, face_dim=128, hidden_dim=64):
        super(ModalAttentionFusion, self).__init__()

        self.v_proj = nn.Linear(voice_dim, hidden_dim)
        self.f_proj = nn.Linear(face_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 动态注意力门控
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        # 融合后的多层感知机分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 0~1 同一个人置信度
        )

    def forward(self, voice_emb, face_emb):
        v_feat = self.relu(self.v_proj(voice_emb))
        f_feat = self.relu(self.f_proj(face_emb))

        concat_feat = torch.cat((v_feat, f_feat), dim=1)
        weights = self.attention_net(concat_feat)  # [batch, 2]

        v_weight = weights[:, 0].unsqueeze(1)
        f_weight = weights[:, 1].unsqueeze(1)

        # 特征乘以各自权重后相加
        fused_feat = (v_feat * v_weight) + (f_feat * f_weight)
        score = self.classifier(fused_feat)
        return score
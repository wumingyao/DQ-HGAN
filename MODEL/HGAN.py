#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : HGAN.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class HGAN(nn.Module):
    """
    这个模型使用了 GATConv 来进行图注意力计算，将对话、意图和情感特征整合到一起进行个性化响应生成。
    其中，使用了三个图卷积层进行特征的更新和传递，使用了三个全连接层进行特征的映射和降维，
    最后使用一个线性层进行情感分类。在模型的前向计算过程中，
    首先对话特征经过一个图卷积层和一个全连接层进行特征提取和映射，
    然后计算对话特征和意图特征之间的注意力权重，并对意图特征进行加权平均。
    接着计算对话特征和情感特征之间的注意力权重，并对情感特征进行加权平均。
    最后将对话特征、加权平均的意图特征和加权平均的情感特征进行拼接，然后通过一个线性层进行情感分类。
    """
    def __init__(self, input_dim, hidden_dim, num_intentions, num_emotions):
        super(HGAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.intention_embedding = nn.Embedding(num_intentions, hidden_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, hidden_dim)
        self.dialogue_gat = GATConv(input_dim, hidden_dim, heads=4, dropout=0.2)
        self.intention_gat = GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.2)
        self.emotion_gat = GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.2)
        self.dialogue_fc = nn.Linear(hidden_dim, hidden_dim)
        self.intention_fc = nn.Linear(hidden_dim, hidden_dim)
        self.emotion_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_emotions)

    def forward(self, dialogue_embeddings, intention_indices, emotion_indices):
        intention_embeddings = self.intention_embedding(intention_indices)
        emotion_embeddings = self.emotion_embedding(emotion_indices)

        dialogue_embeddings = self.dialogue_gat(dialogue_embeddings)
        dialogue_embeddings = F.relu(self.dialogue_fc(dialogue_embeddings))

        intention_embeddings = self.intention_gat(intention_embeddings)
        intention_embeddings = F.relu(self.intention_fc(intention_embeddings))

        emotion_embeddings = self.emotion_gat(emotion_embeddings)
        emotion_embeddings = F.relu(self.emotion_fc(emotion_embeddings))

        intention_attention_weights = F.softmax(torch.matmul(dialogue_embeddings, intention_embeddings.transpose(0, 1)),
                                                dim=-1)
        attended_intentions = torch.matmul(intention_attention_weights.transpose(1, 2), intention_embeddings)

        emotion_attention_weights = F.softmax(torch.matmul(dialogue_embeddings, emotion_embeddings.transpose(0, 1)),
                                              dim=-1)
        attended_emotions = torch.matmul(emotion_attention_weights.transpose(1, 2), emotion_embeddings)

        inputs = torch.cat((dialogue_embeddings, attended_intentions, attended_emotions), dim=-1)
        logits = self.fc(inputs)

        return logits

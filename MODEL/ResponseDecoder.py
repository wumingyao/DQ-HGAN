#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : ResponseDecoder.py


import torch.nn as nn
import torch.nn.functional as F


class ResponseDecoder(nn.Module):
    """
    这个解码器接收到一个策略向量、一个状态向量、一个意图向量和一个情感向量，将这些向量拼接在一起，并通过多层线性变换和激活函数的处理，输出最终的响应向量。
    其中，使用了两个 Embedding 层来分别处理意图和情感信息，使用了三个线性层来映射和降维。在前向计算过程中，首先将策略向量、状态向量、意图向量和情感向量进行拼接，
    然后通过一个线性层和一个 ReLU 激活函数进行特征映射，再通过另一个线性层和 ReLU 激活函数进行特征提取，最后通过一个线性层输出最终的响应向量。
    """
    def __init__(self, hidden_dim, num_intentions, num_emotions):
        super(ResponseDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_intentions = num_intentions
        self.num_emotions = num_emotions

        self.intent_embedding = nn.Embedding(num_intentions, hidden_dim)
        self.emotion_embedding = nn.Embedding(num_emotions, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_emotions)

    def forward(self, policy, state, intention, emotion):
        intent_embed = self.intent_embedding(intention)
        emotion_embed = self.emotion_embedding(emotion)
        state = state.unsqueeze(0)
        inputs = torch.cat([policy, state, intent_embed, emotion_embed], dim=-1)
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
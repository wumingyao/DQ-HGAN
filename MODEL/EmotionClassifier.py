#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : EmotionClassifier.py

import torch.nn as nn
import torch.nn.functional as F


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_emotions):
        super(EmotionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.emotion_embedding = nn.Embedding(num_emotions, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim,
                                                              activation='gelu')
        self.classifier = nn.Linear(hidden_dim, num_emotions)

    def forward(self, dialogue_embeddings, emotion_embeddings):
        attention_weights = F.softmax(torch.matmul(dialogue_embeddings, emotion_embeddings.transpose(0, 1)), dim=-1)
        attended_emotions = torch.matmul(attention_weights.transpose(1, 2), emotion_embeddings)
        inputs = torch.cat((dialogue_embeddings, attended_emotions), dim=-1)
        inputs = inputs.transpose(0, 1)
        encoded = self.transformer_encoder(inputs)
        encoded = encoded.mean(dim=0)
        logits = self.classifier(encoded)
        return logits

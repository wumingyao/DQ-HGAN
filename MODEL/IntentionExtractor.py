#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : IntentionExtractor.py

import torch.nn as nn
import torch.nn.functional as F


class IntentionExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_intentions):
        super(IntentionExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.intention_embedding = nn.Embedding(num_intentions, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim,
                                                              activation='gelu')

    def forward(self, dialogue_embeddings, intention_embeddings):
        attention_weights = F.softmax(torch.matmul(dialogue_embeddings, intention_embeddings.transpose(0, 1)), dim=-1)
        attended_intentions = torch.matmul(attention_weights.transpose(1, 2), intention_embeddings)
        inputs = torch.cat((dialogue_embeddings, attended_intentions), dim=-1)
        inputs = inputs.transpose(0, 1)
        encoded = self.transformer_encoder(inputs)
        encoded = encoded.mean(dim=0)
        return encoded

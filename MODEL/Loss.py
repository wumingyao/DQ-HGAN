#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : Loss.py

import torch.nn.functional as F
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(Loss, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, policy, state, intention, emotion, response, target):
        log_prob = F.log_softmax(response, dim=-1)
        target_prob = F.softmax(target, dim=-1)
        log_prob = log_prob.gather(1, policy.unsqueeze(1))
        loss = -log_prob.mean() + self.weight_decay * (
                    policy.pow(2).sum() + state.pow(2).sum() + intention.pow(2).sum() + emotion.pow(
                2).sum() + response.pow(2).sum() + target.pow(2).sum())
        return loss
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : DialogueEncoder.py

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class DialogueEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(DialogueEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]

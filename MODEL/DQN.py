#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : DQN.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


class DQN:
    def __init__(self, input_dim, hidden_dim, num_intentions, num_emotions, buffer_size=10000, batch_size=32,
                 gamma=0.99, lr=1e-3, update_interval=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_intentions = num_intentions
        self.num_emotions = num_emotions
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_interval = update_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_emotions)
        ).to(self.device)

        self.target_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_emotions)
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.loss_function = nn.MSELoss()

        self.steps = 0

    def select_action(self, state, intention, emotion, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_emotions - 1)
        with torch.no_grad():
            q_values = self.q_network(torch.cat([state, intention, emotion], dim=-1).to(self.device))
            return torch.argmax(q_values).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.FloatTensor([transition[0] for transition in batch]).to(self.device)
        intentions = torch.FloatTensor([transition[1] for transition in batch]).to(self.device)
        emotions = torch.FloatTensor([transition[2] for transition in batch]).to(self.device)
        actions = torch.LongTensor([transition[3] for transition in batch]).to(self.device)
        rewards = torch.FloatTensor([transition[4] for transition in batch]).to(self.device)
        next_states = torch.FloatTensor([transition[5] for transition in batch]).to(self.device)
        next_intentions = torch.FloatTensor([transition[6] for transition in batch]).to(self.device)
        next_emotions = torch.FloatTensor([transition[7] for transition in batch]).to(self.device)
        dones = torch.FloatTensor([transition[8] for transition in batch]).to(self.device)

        q_values = self.q_network(torch.cat([states, intentions, emotions], dim=-1)).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(torch.cat([next_states, next_intentions, next_emotions], dim=-1)).max(1)[
            0].detach()

        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_function(q_values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def add_to_buffer(self, state, intention, emotion, action, reward, next_state, next_intention, next_emotion, done):
        self.replay_buffer.append(
            (state, intention, emotion, action, reward, next_state, next_intention, next_emotion, done))

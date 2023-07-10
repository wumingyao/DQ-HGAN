#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10
# @Author  : 牧礼
# @File    : DQHGAN.py
class MultiSourceEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(MultiSourceEncoder, self).__init__()
        self.dialogue_encoder = nn.TransformerEncoder(...)
        self.intention_extractor = nn.TransformerEncoder(...)
        self.emotion_classifier = nn.TransformerEncoder(...)

    def forward(self, dialogue_history):
        encoded_dialogue = self.dialogue_encoder(dialogue_history)
        intention_feature = self.intention_extractor(dialogue_history)
        emotion_feature = self.emotion_classifier(dialogue_history)
        return encoded_dialogue, intention_feature, emotion_feature


class HeterGraphUserStateTracking(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads):
        super(HeterGraphUserStateTracking, self).__init__()
        self.graph_layer = nn.ModuleList()
        for layer in range(num_layers):
            self.graph_layer.append(HeterogeneousGraphAttention(embedding_dim, num_heads))

    def forward(self, nodes, edges):
        for layer in self.graph_layer:
            nodes = layer(nodes, edges)
        return nodes


class DQNStrategyPlanning(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQNStrategyPlanning, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)
        return q_value


class UtteranceDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(UtteranceDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.output_layer(output)
        return output, hidden


class DQ_HGAN(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, num_layers, num_heads, intention_lexicon,
                 emotion_lexicon):
        super(DQ_HGAN, self).__init__()
        self.encoder = MultiSourceEncoder(hidden_dim)
        self.user_state_tracker = HeterGraphUserStateTracking(embedding_dim, num_layers, num_heads)
        self.strategy_planner = DQNStrategyPlanning(hidden_dim + embedding_dim * 2, hidden_dim, vocab_size)
        self.decoder = UtteranceDecoder(embedding_dim, hidden_dim, vocab_size)
        self.intention_lexicon = intention_lexicon
        self.emotion_lexicon = emotion_lexicon

    def forward(self, dialogue_history):
        encoded_dialogue, intention_feature, emotion_feature = self.encoder(dialogue_history)
        user_state = self.user_state_tracker(nodes, edges)
        strategy_input = torch.cat([encoded_dialogue, intention_feature, emotion_feature], dim=1)
        q_value = self.strategy_planner(strategy_input)
        selected_strategy = torch.argmax(q_value, dim=1)
        decoder_input = selected_strategy.unsqueeze(0)
        decoder_hidden = user_state.unsqueeze(0)
        generated_utterance = []
        for i in range(MAX_UTTERANCE_LENGTH):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            _, topi = output.topk(1)
            decoder_input = topi.squeeze().detach()
            generated_utterance.append(decoder_input.item())
            if decoder_input.item() == EOS_token:
                break
        return generated_utterance

    def calculate_loss(self, dialogue_history, ground_truth):
        generated_utterance = self.forward(dialogue_history)
        loss = nn.NLLLoss()
        nll_loss = loss(torch.tensor(generated_utterance), ground_truth)
        l2_penalty = sum(p.pow(2.0).sum() for p in self.parameters())
        return nll_loss + LAMBDA * l2_penalty

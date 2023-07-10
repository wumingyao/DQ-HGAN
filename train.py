import argparse
import logging
import math
import os
import pickle
import random
import json

import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support,mean_absolute_error
import sklearn.metrics
from transformers.trainer import Trainer
# from clss_trainer import MyTrainer as Trainer
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser
import copy
# from torch.utils.data.dataset import Dataset
from data.Datareader import get_stratege, read_pk, PredictFeedBackDataset
from MODEL.BertModelForFeedBack import BERTMODEL_LIST
from transformers import BertTokenizer
import warnings
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model', default='../MODEL/bert-base-uncased',
                        help='Pretrain model weight')
parser.add_argument('--output_dir', default='./final_output/',
                        help='The output directory where the model predictions and checkpoints will be written.')
parser.add_argument('--data_dir', default='./data/',
                        help='Path saved data')
parser.add_argument('--seed', default=42,
                        help='Path saved data')
parser.add_argument('--per_device_train_batch_size', default=16, type=int)
parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
# parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
parser.add_argument('--source_len', default=512, type=int)
parser.add_argument('--num_train_epochs', default=5, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--lr2', default=5e-5, type=float)
parser.add_argument('--evaluation_strategy', default="epoch", type=str)
parser.add_argument('--save_strategy', default="epoch", type=str)
parser.add_argument('--do_train', default=True)
parser.add_argument('--do_eval', default=True)
parser.add_argument('--do_predict', default=True)
parser.add_argument('--load_best_model_at_end', default=True)
parser.add_argument("--metric_for_best_model", default="b_acc")
parser.add_argument("--model_type", default=6, type=int)
parser.add_argument("--save_total_limit", default=2, type=int)
parser.add_argument("--dataset_type", default=2, type=int)
parser.add_argument("--add_cause", default=False, type=bool)
parser.add_argument("--extend_data", default=1, type=int)
parser.add_argument("--no_origin", default=False, type=bool)
parser.add_argument("--cls", default=False, type=bool)
parser.add_argument("--extend_prefix", default='_beam2', type=str)


# parser.add_argument('--load_best_model_at_end', default=True)
args = parser.parse_args()
print(args.extend_data, args.output_dir)

strateges = get_stratege('../new_strategy.json', norm=True)
stratege2id = {v: k for k, v in enumerate(strateges)}
train_path = args.data_dir + 'train.txt'
val_path = args.data_dir + 'valid.txt'
test_path = args.data_dir + 'test.txt'
tokenizer = BertTokenizer.from_pretrained(args.pretrain_model, use_fast=False)
tokenizer.add_tokens(list(stratege2id.keys()))

Bertmodel = BERTMODEL_LIST[args.model_type]
BertDataset = PredictFeedBackDataset
if args.cls:
    model,loading_info = Bertmodel.from_pretrained(args.pretrain_model, num_labels=6,output_loading_info=True)
else:
    model, loading_info = Bertmodel.from_pretrained(args.pretrain_model, num_labels=1, problem_type="regression",
                                                    output_loading_info=True)
sencond_parameters = loading_info['missing_keys']
model.resize_token_embeddings(len(tokenizer))
if args.extend_data == 1:
    print("we extend data", args.extend_data, type(args.extend_data))
    train_set = BertDataset(train_path, tokenizer, args.source_len, extend_path=f'./final_data/train_extend{args.extend_prefix}.pk', no_origin=args.no_origin,clss=args.cls)
else:
    train_set = BertDataset(train_path, tokenizer, args.source_len, clss=args.cls)
# eval_set = BertDataset(val_path, tokenizer, args.source_len, extend_path=f'./final_data/valid_extend.pk',clss=args.cls)
# test_set = BertDataset(test_path, tokenizer, args.source_len, extend_path=f'./final_data/test_extend.pk',clss=args.cls)
eval_set = BertDataset(val_path, tokenizer, args.source_len,clss=args.cls)
test_set = BertDataset(test_path, tokenizer, args.source_len,clss=args.cls)
print(args.output_dir)

def compute_metrics_with_bart_result(result):
    labels = result.label_ids
    preds = result.predictions

    ##################
    # 计算 predict的指标
    ##################
    if not args.cls:
        mae = mean_absolute_error(labels, preds)
        less_five = 0.
        less_ten = 0.
        less_one = 0.
        t_right, t_pre, t_rr = 1.0, 1.0, 0.
        for l, p in zip(labels, preds):
            if random.random() < 0.01:
                print(f"label={l} and predict={p}")
            if l > 3.0:
                t_right += 1
                if p >= 3.0:
                    t_rr += 1
            if p >= 3.0:
                t_pre += 1
            if abs(l-p) <= 0.5:
                less_five += 1
            if abs(l-p) <= 0.1:
                less_one += 1
            if abs(l-p) <= 1.:
                less_ten += 1
        b_acc = t_rr / t_pre
        b_rec = t_rr / t_right
        dic = {
            "b_acc": b_acc,
            "b_rec": b_rec,
            "b_f1": 2 * b_acc *b_rec / (b_acc + b_rec + 1.0),
            "mean_absolute": mae,
            "less_one": less_one/len(preds),
            "less_five": less_five/len(preds),
            "less_ten": less_ten/len(preds),
        }
    else:
        preds_index = np.argmax(preds, -1)
        preds = preds_index
        dic = {
            "acc": accuracy_score(labels, preds_index),
        }
    return dic

def tmp_socre(result):
    return {"ab": 1.0}

def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

from transformers.optimization import AdamW, Adafactor
def get_optimer(model, second_parameter, train_parser):
    # decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in second_parameter],
            "lr": args.lr2,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in second_parameter],
            "lr": args.learning_rate
        },
    ]
    optimizer_cls = Adafactor if train_parser.adafactor else AdamW
    if train_parser.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (train_parser.adam_beta1, train_parser.adam_beta2),
            "eps": train_parser.adam_epsilon,
        }
    # optimizer_kwargs["lr"] = train_parser.learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def train(self,epochs):
    for epoch in range(epochs):
        dataset = TensorDataset(torch.FloatTensor(self.replay_buffer))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for batch in dataloader:
            states = batch
            dones = batch[:, 8].to(self.device)

            q_values = self.q_network(torch.cat([states, intentions.unsqueeze(-1), emotions.unsqueeze(-1)], dim=-1)).gather(1,
                                                                                                                            actions.unsqueeze(
                                                                                                                                1))
            next_q_values = self.target_network(
                torch.cat([next_states, next_intentions.unsqueeze(-1), next_emotions.unsqueeze(-1)], dim=-1)).max(1)[0].detach()

            targets = rewards + (1 - dones) * self.gamma * next_q_values

            loss = self.loss_function(q_values, targets.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    self.steps += 1
    if self.steps % self.update_interval == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_hgan(self, dataset, tokenizer, model):
    hgan = HierarchicalGAN(self.input_dim, self.hidden_dim, self.num_intentions, self.num_emotions, tokenizer, model,
                           device=self.device)

    for epoch in range(hgan.num_epochs):
        hgan.train()
        for batch in dataset:
            batch = batch.to(self.device)
            states = batch[:, :self.input_dim]
            intentions = batch[:, self.input_dim:self.input_dim + self.num_intentions]
            emotions = batch[:,
                       self.input_dim + self.num_intentions:self.input_dim + self.num_intentions + self.num_emotions]
            inputs = batch[:, -1]
            hgan.train_batch(states, intentions, emotions, inputs)

        hgan.eval()
        with torch.no_grad():
            for batch in dataset:
                batch = batch.to(self.device)
                states = batch[:, :self.input_dim]
                intentions = batch[:, self.input_dim:self.input_dim + self.num_intentions]
                emotions = batch[:,
                           self.input_dim + self.num_intentions:self.input_dim + self.num_intentions + self.num_emotions]
                inputs = batch[:, -1]
                hgan.eval_batch(states, intentions, emotions, inputs)


def train_dq_hgan(self, num_episodes, max_episode_length, epsilon_start, epsilon_final, epsilon_decay, bert_model_name,
                  gpt2_model_name):
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(self.device)
    self.train_hgan(self.replay_buffer, bert_tokenizer, gpt2_model)

    epsilon = epsilon_start
    for episode in range(num_episodes):
        state = torch.zeros(self.input_dim).to(self.device)
        intention = torch.zeros(self.num_intentions).long().to(self.device)
        emotion = torch.zeros(self.num_emotions).long().to(self.device)
        cum_reward = 0
        for t in range(max_episode_length):
            action = self.select_action(state, intention, emotion, epsilon)
            next_state, reward, done, next_intention, next_emotion = self.env.step(action)

            self.add_to_buffer(state, intention, emotion, action, reward, next_state, next_intention, next_emotion,
                               done)
            self.update()

            state = next_state
            intention = next_intention
            emotion = next_emotion
            cum_reward += reward

            if done:
                break

        epsilon = max(epsilon_final, epsilon_decay * epsilon)

        if episode % 10 == 0:
            print(f"Episode {episode}: Cumulative reward = {cum_reward}")

        if episode % 100 == 0:
            self.train_hgan(self.replay_buffer, bert_tokenizer, gpt2_model)

def generate_response(state, intention, emotion, dqn, hgan, bert_tokenizer, gpt2_tokenizer, gpt2_model, max_length=50):
    with torch.no_grad():
        inputs = hgan.generate(state, intention, emotion, bert_tokenizer, gpt2_tokenizer, gpt2_model, max_length=max_length)
        inputs = torch.FloatTensor(inputs).to(dqn.device)
        q_values = dqn.q_network(torch.cat([state, intention, emotion], dim=-1).to(dqn.device) + inputs)
        action = torch.argmax(q_values).item()
        response = hgan.generate(state, intention, action, bert_tokenizer, gpt2_tokenizer, gpt2_model, max_length=max_length)
        return response
if __name__ == '__main__':
    os.environ["WANDB_DISABLED"] = "true"
    fix_random(args.seed)
    train()
    # generate_response(state, intention, emotion, dqn, hgan, bert_tokenizer, gpt2_tokenizer, gpt2_model)

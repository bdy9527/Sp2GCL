import os
import yaml
import time
import math
import copy
import random
import argparse
import numpy as np
import pandas as pd
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from graph_dataset_pyg import PygGraphPropPredDataset
from utils import init_params, seed_everything, collate_basis, collate_basis_sign
from model import MoleculeEncoder, EigenMLP_BN, SpaSpeGraph
from evaluation import ogbg_evaluation
from ogb.graphproppred import Evaluator


def train_epoch(model, train_loader, optimizer):
    model.train()

    batch_loss = 0.
    batch_acc = 0.
    for (spa_batch, spe_batch) in train_loader:
        spa_batch = spa_batch.to(device)
        spe_batch = spe_batch.to(device)

        optimizer.zero_grad()
        loss = model(spa_batch, spe_batch)
        batch_loss += loss.item()

        loss.backward()
        optimizer.step()

    return batch_loss / len(train_loader) 


def eval_epoch(model, valid_loader):
    model.eval()

    emb = []
    y_true = []

    for (spa_batch, spe_batch) in valid_loader:
        spa_batch = spa_batch.to(device)
        spe_batch = spe_batch.to(device)

        with torch.no_grad():
            _, spa = model.spa_encoder(spa_batch.batch, spa_batch.x, spa_batch.edge_index, spa_batch.edge_attr)
            _, spe = model.spe_encoder(spe_batch.batch, spe_batch.x, spe_batch.edge_index, spe_batch.edge_attr)
        
        spa, spe = spa.detach().cpu(), spe.detach().cpu()
        emb.append((spa + spe) / 2.)
        y_true.append(spa_batch.y.detach().cpu())

    emb = torch.cat(emb, dim=0).numpy()
    y_true = torch.cat(y_true, dim=0).numpy()
    
    return emb, y_true


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='molesol')
    args = parser.parse_args()

    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    print(config)

    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)

    dataset_name = 'ogbg-{}'.format(args.dataset)
    
    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    bs = config['batch_size']
    period = config['period']
    hidden_dim = config['hidden_dim']
    pooling = config['pooling']
    spa_layer = config['spa_layer']
    spe_layer = config['spe_layer']
    spa_dropout = config['spa_dropout']
    spe_dropout = config['spe_dropout']
    t = config['t']

    dataset = PygGraphPropPredDataset(name = dataset_name)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(dataset_name)
    task_type = dataset.task_type
    num_tasks = dataset.num_tasks

    train_loader = DataLoader(dataset, batch_size=bs, shuffle=True,  collate_fn=partial(collate_basis_sign, period=period))
    valid_loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=partial(collate_basis_sign, period=period))
    #train_loader = DataLoader(dataset, batch_size=bs, shuffle=True,  collate_fn=partial(collate_pyg, period=period))
    #valid_loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=partial(collate_pyg, period=period))

    spa_encoder = MoleculeEncoder(hidden_dim, spa_layer, spa_dropout, pooling)
    spe_encoder = EigenMLP_BN(period, hidden_dim, spe_layer, spe_dropout, pooling)
    model = SpaSpeGraph(spa_encoder, spe_encoder, hidden_dim, t).to(device)

    evaluator = Evaluator(dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    valid_curve = []
    test_curve = []
    train_curve = []

    print('training')

    emb, y_true = eval_epoch(model, valid_loader)
    train_score, val_score, test_score = ogbg_evaluation(emb, y_true, split_idx, evaluator, task_type, num_tasks)

    print('Before training: ', train_score, val_score, test_score)

    for i in range(epoch):
        model.train()

        batch_loss = 0.
        batch_acc = 0.
        for (spa_batch, spe_batch) in train_loader:
            spa_batch = spa_batch.to(device)
            spe_batch = spe_batch.to(device)

            optimizer.zero_grad()
            loss = model(spa_batch, spe_batch)
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()

        emb, y_true = eval_epoch(model, valid_loader)
        train_score, val_score, test_score = ogbg_evaluation(emb, y_true, split_idx, evaluator, task_type, num_tasks)

        train_curve.append(train_score)
        valid_curve.append(val_score)
        test_curve.append(test_score)

        print(batch_loss/len(train_loader), train_score, val_score, test_score)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('final: ', best_val_epoch, valid_curve[best_val_epoch], test_curve[best_val_epoch])


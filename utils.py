import torch
import torch.nn as nn
import random
import os
import math
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_auc_score, average_precision_score


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split(y):

    train_ratio = 0.1
    val_ratio = 0.1
    test_ratio = 0.8

    N = len(y)
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)

    return train_idx, val_idx, test_idx 


def collate_basis(graphs, period):
    graph_list = []

    for g in graphs:
        num_nodes = g.num_nodes

        e = g.e
        u = g.u.view(num_nodes, num_nodes)

        period_term = torch.arange(period, device=u.device, dtype=torch.float32)
        period_e = e.unsqueeze(1) * period_term
        fourier_e = torch.cat([torch.sin(period_e), torch.cos(period_e)], dim=-1)
        equ = u @ fourier_e

        new_g = Data()
        new_g.num_nodes = g.num_nodes
        new_g.x = g.x
        new_g.y = g.y
        new_g.pos = equ
        new_g.edge_index = g.edge_index
        new_g.edge_attr = g.edge_attr

        graph_list.append(new_g)

    batched_graph = Batch.from_data_list(graph_list)

    return batched_graph


def collate_basis_sign(graphs, period):
    spa_graph_list = []
    spe_graph_list = []

    for g in graphs:
        num_nodes = g.num_nodes

        spa_g = Data()
        spa_g.num_nodes = g.num_nodes
        spa_g.x = g.x
        spa_g.y = g.y
        spa_g.edge_index = g.edge_index
        spa_g.edge_attr = g.edge_attr

        e = g.e
        period_term = torch.arange(period, device=e.device, dtype=torch.float32)
        period_e = e.unsqueeze(1) * period_term
        fourier_e = torch.cat([torch.sin(period_e), torch.cos(period_e)], dim=-1)

        udj = g.u.view(num_nodes, num_nodes)
        row, col = udj.nonzero().t()

        spe_g = Data()
        spe_g.num_nodes = g.num_nodes
        spe_g.x = fourier_e
        spe_g.edge_index = torch.stack([col, row], dim=0)
        spe_g.edge_attr = udj[row, col]                      # to align with the source-to-target direction

        spa_graph_list.append(spa_g)
        spe_graph_list.append(spe_g)

    spa_batched_graph = Batch.from_data_list(spa_graph_list)
    spe_batched_graph = Batch.from_data_list(spe_graph_list)

    return spa_batched_graph, spe_batched_graph



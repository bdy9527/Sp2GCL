import os
import time
import math
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_params, seed_everything, split

from evaluation import node_evaluation
from model import GCN, EigenNeuron, EigenMLP, SpaSpeNode, Encoder, Basic, SAN


def main(args):
    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)

    if args.dataset in ['pubmed', 'flickr', 'arxiv', 'wiki', 'facebook']:
        data = torch.load('data/{}.pt'.format(args.dataset))

        x = data.x.float().to(device)
        edge = data.edge_index.long().to(device)
        e = data.e[:args.spe_dim].float().to(device)
        u = data.u[:, :args.spe_dim].float().to(device)

        y = data.y
        print(y.min().item(), y.max().item())
        if 'train_mask' in data.keys:
            if len(data.train_mask.size()) > 1:
                train_idx = torch.where(data.train_mask[:, args.seed])[0]
                val_idx = torch.where(data.val_mask[:, args.seed])[0]
                test_idx = torch.where(data.test_mask)[0]
            else:
                train_idx = torch.where(data.train_mask)[0]
                val_idx = torch.where(data.val_mask)[0]
                test_idx = torch.where(data.test_mask)[0]
        else:
            train_idx, val_idx, test_idx = split(y)

    else:
        pass

    print(len(test_idx))

    #spa_encoder = GCN(x.size(1), args.hidden_dim, args.hidden_dim).to(device)
    spa_encoder = Encoder(x.size(1), args.hidden_dim, args.hidden_dim).to(device)
    spe_encoder = EigenMLP(args.spe_dim, args.hidden_dim, args.hidden_dim, args.period).to(device)
    #spe_encoder = Basic(args.spe_dim, args.hidden_dim, args.hidden_dim).to(device)
    #spe_encoder = SAN(args.spe_dim, args.hidden_dim, args.hidden_dim).to(device)

    model = SpaSpeNode(spa_encoder, spe_encoder, args.hidden_dim, args.t).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    '''
    flip = 2 * torch.randint(0, 2, (args.spe_dim,)) - 1
    sign_flip = torch.diag(flip).float().to(device)
    coor_flip = torch.randperm(args.spe_dim).to(device)

    uuu = torch.mm(u, sign_flip)
    uu = u.clone()[:, coor_flip]
    ee = e.clone()[coor_flip]
    '''

    t1 = time.time()
    for i in range(1000):
        model.eval()
        spe_emb = model.spe_encoder(e, u).detach()
    t2 = time.time()
    print(t2 - t1)

    for idx in range(100):
        model.train()
        loss = model(x, edge, e, u)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (idx+1) % 10 == 0:
            model.eval()

            spa_emb = model.spa_encoder(x, edge).detach()
            spe_emb = model.spe_encoder(e, u).detach()

            acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
            #acc, pred = node_evaluation(torch.cat((spa_emb, spe_emb), dim=-1), y, train_idx, val_idx, test_idx)
            #acc, pred = node_evaluation(spe_emb, y, train_idx, val_idx, test_idx)

            print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--dataset', default='facebook')

    parser.add_argument('--spe_dim', type=int, default=100)
    parser.add_argument('--period', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--t', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    print(args)

    main(args)


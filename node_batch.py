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
from model import GCN, EigenNeuron, EigenMLP, SpaSpeNode, Encoder
from torch_geometric.loader import NeighborLoader



def main(args):
    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)

    if args.dataset in ['pubmed', 'computer', 'photo', 'wiki', 'cs', 'physics']:
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
            print('self split')
            train_idx, val_idx, test_idx = split(y)

    elif args.dataset in ['flickr', 'arxiv']:
        data = torch.load('data/{}.pt'.format(args.dataset))

        x = data.x
        y = data.y

        train_idx = torch.where(data.train_mask)[0]
        val_idx = torch.where(data.val_mask)[0]
        test_idx = torch.where(data.test_mask)[0]

        train_loader = NeighborLoader(data, batch_size=2048, num_neighbors=[-1, -1], shuffle=True)
        infer_loader = NeighborLoader(data, batch_size=2048, num_neighbors=[-1, -1], shuffle=False)

    print(len(test_idx))

    spa_encoder = GCN(x.size(1), args.hidden_dim, args.hidden_dim).to(device)
    #spa_encoder = Encoder(x.size(1), args.hidden_dim, args.hidden_dim).to(device)
    spe_encoder = EigenMLP(args.spe_dim, args.hidden_dim, args.period).to(device)

    model = SpaSpeNode(spa_encoder, spe_encoder, args.hidden_dim, args.t).to(device)
    model.apply(init_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    record = []
    for idx in range(1000):
        model.train()

        for batch in train_loader:
            x = batch.x.to(device)
            edge = batch.edge_index.to(device)
            e = batch.e.to(device)
            u = batch.u[:batch.batch_size].to(device)

            loss = model(x, edge, e, u, batch.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (idx+1) % 1 == 0:
            model.eval()
                    
            spa_emb = []
            spe_emb = []
        
            for batch in infer_loader:
                x = batch.x.to(device)
                edge = batch.edge_index.to(device)
                e = batch.e.to(device)
                u = batch.u[:batch.batch_size].to(device)
            
                h_a = model.spa_encoder(x, edge)[:batch.batch_size, :]
                h_e = model.spe_encoder(e, u)
            
                spa_emb.append(h_a.detach())
                spe_emb.append(h_e.detach())
        
            spa_emb = torch.cat(spa_emb, dim=0)
            spe_emb = torch.cat(spe_emb, dim=0)
        
            acc, pred = node_evaluation((spa_emb + spe_emb)/2, y, train_idx, val_idx, test_idx)
            record.append(acc)

            print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', default='flickr')

    parser.add_argument('--spe_dim', type=int, default=500)
    parser.add_argument('--period', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--t', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    print(args)

    main(args)


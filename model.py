import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GINEConv, GINConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm


class EigenConv(MessagePassing):
    def __init__(self, NN, edge_mlp_dim=16, sign=False):
        super(EigenConv, self).__init__()

        self.nn = NN
        self.sign = sign

        self.phi = nn.Sequential(nn.Linear(1, edge_mlp_dim), nn.ReLU(), nn.Linear(edge_mlp_dim, edge_mlp_dim))
        self.psi = nn.Sequential(nn.Linear(edge_mlp_dim, edge_mlp_dim), nn.ReLU(), nn.Linear(edge_mlp_dim, 1))

    def forward(self, x, edge_index, edge_weight):

        h = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.nn(h)

    def message(self, x_j, edge_weight):
        edge_weight = edge_weight.view(-1, 1)
        if self.sign:
            sign_inv_u = self.psi(self.phi(edge_weight) + self.phi(-edge_weight))
        else:
            sign_inv_u = edge_weight
        return sign_inv_u * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels, momentum = 0.01)
        self.prelu1 = nn.PReLU()
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum = 0.01)
        self.prelu2 = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(self.bn2(x))

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x


class Basic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Basic, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x


class SAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SAN, self).__init__()

        self.k = input_dim
        self.linear = nn.Linear(2, 16)
        self.mha = nn.MultiheadAttention(16, num_heads=1, batch_first=True)
        self.ffn = MLP(16, hidden_dim, hidden_dim)

        self.mha_norm = nn.LayerNorm(16)
        self.ffn_norm = nn.LayerNorm(16)

    def forward(self, e, u):
        # e: [k]
        # u: [N, k]
        N = u.size(0)

        e = e.expand(N, -1)
        h = torch.cat((e.unsqueeze(2), u.unsqueeze(2)), dim=2)  # [N, k, 2]
        h = self.linear(h)                                      # [N, k, d]

        mha_h = self.mha_norm(h)
        h, attn = self.mha(mha_h, mha_h, mha_h)

        ffn_h = self.ffn_norm(h)
        h = self.ffn(ffn_h)

        h = torch.sum(h, dim=1)

        return h


class MoleculeEncoder(nn.Module):
    def __init__(self, hidden_dim=300, nlayer=5, dropout=0., pooling='add'):
        super(MoleculeEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.dropout = dropout
        self.pooling = pooling

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        self.CNs = nn.ModuleList()
        self.BNs = nn.ModuleList()

        for i in range(nlayer):
            MLP = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), 
                               nn.BatchNorm1d(2*hidden_dim), 
                               nn.ReLU(), 
                               nn.Linear(2*hidden_dim, hidden_dim))
            conv = GINEConv(MLP)
            bn = nn.BatchNorm1d(hidden_dim)
            self.CNs.append(conv)
            self.BNs.append(bn)

        self.reset_parameter()

    def reset_parameter(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr):

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        for i in range(self.nlayer):
            x = self.CNs[i](x, edge_index, edge_attr)
            x = self.BNs[i](x)
            if i == self.nlayer - 1:
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)

        xpool = global_add_pool(x, batch)

        return x, xpool


class EigenMLP_BN(nn.Module):
    def __init__(self, period=10, hidden_dim=300, nlayer=5, dropout=0., pooling='add'):
        super(EigenMLP_BN, self).__init__()

        self.nlayer = nlayer
        self.dropout = dropout

        self.linear = nn.Linear(2*period, hidden_dim)
        self.CNs = nn.ModuleList()
        self.BNs = nn.ModuleList()

        for i in range(nlayer):
            MLP = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), 
                                nn.BatchNorm1d(2*hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(2*hidden_dim, hidden_dim))
            conv = EigenConv(MLP)
            bn = nn.BatchNorm1d(hidden_dim)
            self.CNs.append(conv)
            self.BNs.append(bn)

        self.reset_parameter()

    def reset_parameter(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr):

        x = self.linear(x)

        for i in range(self.nlayer):
            x = self.CNs[i](x, edge_index, edge_attr)
            x = self.BNs[i](x)
            if i == self.nlayer - 1:
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)

        xpool = global_add_pool(x, batch)

        return x, xpool


class EigenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, period):
        super(EigenMLP, self).__init__()

        self.k = input_dim
        self.period = period

        self.phi = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16))
        self.psi = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

        self.mlp1 = nn.Linear(2*period, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, e, u):
        e = e * 100

        u = u.unsqueeze(2)
        # u = self.psi(self.phi(u) + self.phi(-u)).squeeze(2)
        # u = (self.phi(u) + self.phi(-u)).squeeze(2)

        period_term = torch.arange(0, self.period, device=u.device).float()
        period_e = e.unsqueeze(1) * torch.pow(2, period_term) 
        fourier_e = torch.cat([torch.sin(period_e), torch.cos(period_e)], dim=-1)  # [k, 2f]

        h = u @ fourier_e
        h = self.mlp1(h)
        h = F.relu(h)
        h = self.mlp2(h)

        return h


class SpaSpeNode(nn.Module):
    def __init__(self, spa_encoder, spe_encoder, hidden_dim, t=1.):
        super(SpaSpeNode, self).__init__()

        self.t = t
        self.hidden_dim = hidden_dim

        self.spa_encoder = spa_encoder
        self.spe_encoder = spe_encoder

        #self.spa_node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        #self.spe_node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, e, u, size=-1):
        x_node_spa = self.spa_encoder(x, edge_index)
        x_node_spe = self.spe_encoder(e, u)

        if size > 0:
            x_node_spa = x_node_spa[:size, :]

        #h_node_spa = self.spa_node_proj(x_node_spa)
        #h_node_spe = self.spe_node_proj(x_node_spe)
        h_node_spa = self.proj(x_node_spa)
        h_node_spe = self.proj(x_node_spe)

        return CLIP(h_node_spa, h_node_spe, t=self.t)


class SpaSpeGraph(nn.Module):
    def __init__(self, spa_encoder, spe_encoder, hidden_dim, t=1.):
        super(SpaSpeGraph, self).__init__()

        self.t = t
        self.hidden_dim = hidden_dim

        self.spa_encoder = spa_encoder
        self.spe_encoder = spe_encoder

        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, spa_batch, spe_batch):
        x_node_a, x_graph_a = self.spa_encoder(spa_batch.batch, spa_batch.x, spa_batch.edge_index, spa_batch.edge_attr)
        x_node_e, x_graph_e = self.spe_encoder(spe_batch.batch, spe_batch.x, spe_batch.edge_index, spe_batch.edge_attr)

        h_graph_a = self.proj(x_graph_a)
        h_graph_e = self.proj(x_graph_e)

        return CLIP(h_graph_a, h_graph_e, t=self.t)


def CLIP(h1, h2, t=1.):
    h1 = F.normalize(h1, dim=-1, p=2)
    h2 = F.normalize(h2, dim=-1, p=2)

    logits = torch.mm(h1, h2.t()) / t
    labels = torch.arange(h1.size(0), device=h1.device, dtype=torch.long)

    return 0.5 * F.cross_entropy(logits, labels) + 0.5 * F.cross_entropy(logits.t(), labels)


from __future__ import division

import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json

import editdistance

import numpy as np
import six
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device


class GCN(torch.nn.Module):
    def __init__(self, embdim, treehid, nlayer, dropout, residual=False, tied=False, nhead=1, edgedrop=0.0):
        super(GCN, self).__init__()
        self.treehid = treehid
        self.residual = residual
        self.embdim = embdim
        self.tied = tied
        self.nhead = nhead
        self.edgedrop = edgedrop
        if self.tied:
            assert treehid == embdim
        self.gcn_l1 = torch.nn.Linear(self.embdim, self.treehid)
        if self.residual:
            self.layernorm_l1 = torch.nn.LayerNorm(self.treehid)
        for i in range(nlayer-1):
            setattr(self, 'gcn_l{}'.format(i+2), torch.nn.Linear(self.treehid, self.treehid))
            if self.tied and i < nlayer - 2:
                getattr(self, 'gcn_l{}'.format(i+2)).weight = self.gcn_l1.weight
                getattr(self, 'gcn_l{}'.format(i+2)).bias = self.gcn_l1.bias
            setattr(self, 'layernorm_l{}'.format(i+2), torch.nn.LayerNorm(self.treehid))
        self.dropout = torch.nn.Dropout(dropout)
        self.nlayer = nlayer

    def get_lextree_encs_gcn(self, decemb, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[0] == {} and wordpiece is not None:
            idx = len(embeddings)
            ey = decemb.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                ey = decemb.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs_gcn(decemb, values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx
        else:
            import pdb; pdb.set_trace()

    def forward_gcn(self, lextree, embeddings, adjacency):
        n_nodes = len(embeddings)
        nodes_encs = torch.cat(embeddings, dim=0)
        adjacency_mat = nodes_encs.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)

        if self.training:
            edgedropmat = torch.rand(adjacency_mat.size()).to(adjacency_mat.device) < self.edgedrop
            adjacency_mat = adjacency_mat.masked_fill(edgedropmat, 0.0)
        all_node_encs = []
        for i in range(self.nlayer):
            next_nodes_encs = getattr(self, 'gcn_l{}'.format(i+1))(nodes_encs)
            if i == 0:
                first_layer_enc = next_nodes_encs
            if self.nhead > 1 and i == 0:
                all_node_encs.append(next_nodes_encs)
            next_nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, next_nodes_encs))
            if self.residual and i > 0:
                nodes_encs = next_nodes_encs + nodes_encs
                nodes_encs = getattr(self, 'layernorm_l{}'.format(i+1))(nodes_encs)
            elif self.residual:
                nodes_encs = next_nodes_encs + first_layer_enc
                nodes_encs = getattr(self, 'layernorm_l{}'.format(i+1))(nodes_encs)
            else:
                nodes_encs = next_nodes_encs
            all_node_encs.append(nodes_encs)
        if self.nhead > 1:
            output_encs = all_node_encs[self.nlayer//2:self.nlayer//2+1] + all_node_encs[-self.nhead+1:]
            nodes_encs = torch.cat(output_encs, dim=-1)
        return nodes_encs

    def fill_lextree_encs_gcn(self, lextree, nodes_encs, wordpiece=None):
        if lextree[0] == {} and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs_gcn(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        else:
            import pdb; pdb.set_trace()

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs_gcn(decemb, prefixtree, embeddings, adjacency)
        nodes_encs = self.forward_gcn(prefixtree, embeddings, adjacency)
        self.fill_lextree_encs_gcn(prefixtree, nodes_encs)


class GCNSage(torch.nn.Module):
    def __init__(self, embdim, treehid, nlayer, dropout, residual=False, tied=False, nhead=1):
        super(GCNSage, self).__init__()
        self.treehid = treehid
        self.residual = residual
        self.embdim = embdim
        self.tied = tied
        self.nhead = nhead
        if self.tied:
            assert treehid == embdim
        self.gcn_l1 = torch.nn.Linear(self.embdim, self.treehid)
        self.sage_pool_1 = torch.nn.Linear(self.embdim, self.treehid)
        self.sage_merge_1 = torch.nn.Linear(self.embdim + self.treehid, self.treehid)
        if self.residual:
            self.gcn_layernorm_l1 = torch.nn.LayerNorm(self.treehid)
        for i in range(nlayer-1):
            # GCN layers
            setattr(self, 'gcn_l{}'.format(i+2), torch.nn.Linear(self.treehid, self.treehid))
            if self.tied:
                getattr(self, 'gcn_l{}'.format(i+2)).weight = self.gcn_l1.weight
                getattr(self, 'gcn_l{}'.format(i+2)).bias = self.gcn_l1.bias
            setattr(self, 'gcn_layernorm_l{}'.format(i+2), torch.nn.LayerNorm(self.treehid))
            # Sage layers
            setattr(self, 'sage_pool_{}'.format(i+2), torch.nn.Linear(self.treehid, self.treehid))
            setattr(self, 'sage_merge_{}'.format(i+2), torch.nn.Linear(self.treehid * 2, self.treehid))
            if self.tied:
                assert self.embdim == self.treehid
                getattr(self, 'sage_pool_{}'.format(i+2)).weight = self.sage_pool_1.weight
                getattr(self, 'sage_pool_{}'.format(i+2)).bias = self.sage_pool_1.bias
                getattr(self, 'sage_merge_{}'.format(i+2)).weight = self.sage_merge_1.weight
                getattr(self, 'sage_merge_{}'.format(i+2)).bias = self.sage_merge_1.bias
            if self.residual:
                setattr(self, 'sage_layernorm_l{}'.format(i+2), torch.nn.LayerNorm(self.treehid))
        self.dropout = torch.nn.Dropout(dropout)
        self.nlayer = nlayer

    def get_lextree_encs(self, decemb, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = len(embeddings)
            ey = decemb.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[1] == -1 and lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                # ey = self.dec.embed(to_device(self, torch.LongTensor([wordpiece])))
                ey = decemb.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs(decemb, values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def forward_sage_layer(self, embeddings, adjacency, layerid=1):
        if layerid == 1:
            nodes_encs = torch.relu(getattr(self, 'sage_pool_{}'.format(layerid))(embeddings))
        else:
            nodes_encs = embeddings
        pooled_encs = [0] * len(adjacency)
        for i, node in enumerate(adjacency):
            # node = torch.nonzero(node).squeeze(1).tolist()
            if len(node) > 1:
                candidates = nodes_encs[node[1:]]
                pooled_encs[node[0]] = candidates.max(dim=0)[0]
            elif len(node) == 1:
                pooled_encs[node[0]] = nodes_encs.new_zeros(nodes_encs.size(1))
        pooled_encs = torch.stack(pooled_encs)

        pooled_encs = torch.cat([embeddings, pooled_encs], dim=1)
        pooled_encs = torch.relu(getattr(self, 'sage_merge_{}'.format(layerid))(pooled_encs))
        return pooled_encs

    def forward_gcn(self, embeddings, adjacency):
        n_nodes = len(embeddings)
        nodes_encs = torch.cat(embeddings, dim=0)
        adjacency_mat = nodes_encs.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)

        all_node_encs = []
        for i in range(self.nlayer):
            next_nodes_encs = getattr(self, 'gcn_l{}'.format(i+1))(nodes_encs)
            if i == 0:
                first_layer_enc = next_nodes_encs
            if self.nhead > 1 and i == 0:
                all_node_encs.append(next_nodes_encs)
            next_nodes_encs = torch.relu(torch.einsum('ij,jk->ik', adjacency_mat, next_nodes_encs))
            if self.residual and i > 0:
                nodes_encs = next_nodes_encs + nodes_encs
                nodes_encs = getattr(self, 'gcn_layernorm_l{}'.format(i+1))(nodes_encs)
            elif self.residual:
                nodes_encs = next_nodes_encs + first_layer_enc
                nodes_encs = getattr(self, 'gcn_layernorm_l{}'.format(i+1))(nodes_encs)
            else:
                nodes_encs = next_nodes_encs
            all_node_encs.append(nodes_encs)
        if self.nhead > 1:
            output_encs = all_node_encs[0:1] + all_node_encs[-self.nhead+1:]
            nodes_encs = torch.cat(output_encs, dim=-1)
        return nodes_encs

    def forward_sage(self, embeddings, adjacency):
        embeddings = torch.cat(embeddings, dim=0)
        sage_nodes_encs = self.forward_sage_layer(embeddings, adjacency)
        all_node_encs = [sage_nodes_encs]
        for i in range(self.nlayer-1):
            next_nodes_encs = self.forward_sage_layer(sage_nodes_encs, adjacency, layerid=i+2)
            if self.residual:
                sage_nodes_encs = next_nodes_encs + sage_nodes_encs
            else:
                sage_nodes_encs = next_nodes_encs
            sage_nodes_encs = getattr(self, 'sage_layernorm_l{}'.format(i+2))(sage_nodes_encs)
            all_node_encs.append(sage_nodes_encs)
        if self.nhead > 1:
            output_encs = all_node_encs[0:1] + all_node_encs[-self.nhead+1:]
            sage_nodes_encs = torch.cat(output_encs, dim=-1)
        return sage_nodes_encs

    def fill_lextree_encs(self, lextree, nodes_encs, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[1] == -1 and lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs(decemb, prefixtree, embeddings, adjacency)
        # Forward GCN
        gcn_nodes_encs = self.forward_gcn(embeddings, adjacency)
        # Forward Sage
        sage_nodes_encs = self.forward_sage(embeddings, adjacency)
        # Combine both
        nodes_encs = torch.cat([gcn_nodes_encs, sage_nodes_encs], dim=-1)
        self.fill_lextree_encs(prefixtree, nodes_encs)


class GSage(torch.nn.Module):
    def __init__(self, embdim, treehid, nlayer, dropout, residual=False, tied=False, nhead=1):
        super(GSage, self).__init__()
        self.treehid = treehid
        self.residual = residual
        self.embdim = embdim
        self.tied = tied
        self.sage_pool_1 = torch.nn.Linear(self.embdim, self.treehid)
        self.sage_merge_1 = torch.nn.Linear(self.embdim + self.treehid, self.treehid)
        self.nhead = nhead
        for i in range(nlayer-1):
            setattr(self, 'sage_pool_{}'.format(i+2), torch.nn.Linear(self.treehid, self.treehid))
            setattr(self, 'sage_merge_{}'.format(i+2), torch.nn.Linear(self.treehid * 2, self.treehid))
            if self.tied:
                assert self.embdim == self.treehid
                getattr(self, 'sage_pool_{}'.format(i+2)).weight = self.sage_pool_1.weight
                getattr(self, 'sage_pool_{}'.format(i+2)).bias = self.sage_pool_1.bias
                getattr(self, 'sage_merge_{}'.format(i+2)).weight = self.sage_merge_1.weight
                getattr(self, 'sage_merge_{}'.format(i+2)).bias = self.sage_merge_1.bias
            if self.residual:
                setattr(self, 'layernorm_l{}'.format(i+2), torch.nn.LayerNorm(self.treehid))
        self.dropout = torch.nn.Dropout(dropout)
        self.nlayer = nlayer

    def get_lextree_encs_sage(self, decemb, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = len(embeddings)
            ey = decemb.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[1] == -1 and lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                ey = decemb.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs_sage(decemb, values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def forward_sage(self, embeddings, adjacency, layerid=1):
        if layerid == 1:
            nodes_encs = torch.relu(getattr(self, 'sage_pool_{}'.format(layerid))(embeddings))
        else:
            nodes_encs = embeddings
        pooled_encs = [0] * len(adjacency)
        for i, node in enumerate(adjacency):
            # node = torch.nonzero(node).squeeze(1).tolist()
            if len(node) > 1:
                candidates = nodes_encs[node[1:]]
                pooled_encs[node[0]] = candidates.max(dim=0)[0]
            elif len(node) == 1:
                pooled_encs[node[0]] = nodes_encs.new_zeros(nodes_encs.size(1))
        pooled_encs = torch.stack(pooled_encs)

        pooled_encs = torch.cat([embeddings, pooled_encs], dim=1)
        pooled_encs = torch.relu(getattr(self, 'sage_merge_{}'.format(layerid))(pooled_encs))
        return pooled_encs

    def fill_lextree_encs_sage(self, lextree, nodes_encs, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[1] == -1 and lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs_sage(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs_sage(decemb, prefixtree, embeddings, adjacency)
        embeddings = torch.cat(embeddings, dim=0)
        all_node_encs = []

        nodes_encs = self.forward_sage(embeddings, adjacency)
        all_node_encs.append(nodes_encs)
        for i in range(self.nlayer-1):
            next_nodes_encs = self.forward_sage(nodes_encs, adjacency, layerid=i+2)
            if self.residual:
                nodes_encs = next_nodes_encs + nodes_encs
            else:
                nodes_encs = next_nodes_encs
            nodes_encs = getattr(self, 'layernorm_l{}'.format(i+2))(nodes_encs)
            all_node_encs.append(nodes_encs)
        if self.nhead > 1:
            output_encs = all_node_encs[0:1] + all_node_encs[-self.nhead+1:]
            nodes_encs = torch.cat(output_encs, dim=-1)
        self.fill_lextree_encs_sage(prefixtree, nodes_encs)


class SageMean(GSage):
    def __init__(self, embdim, treehid, nlayer, dropout, residual=False):
        super(SageMean, self).__init__(embdim, treehid, nlayer, dropout, residual)

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs_sage(decemb, prefixtree, embeddings, adjacency)
        embeddings = torch.cat(embeddings, dim=0)
        n_nodes = len(embeddings)
        adjacency_mat = embeddings.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0

        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)

        nodes_encs = self.forward_sage(embeddings, adjacency_mat)
        for i in range(self.nlayer-1):
            next_nodes_encs = self.forward_sage(nodes_encs, adjacency_mat, layerid=i+2)
            if self.residual:
                nodes_encs = next_nodes_encs + nodes_encs
            else:
                nodes_encs = next_nodes_encs
            nodes_encs = getattr(self, 'layernorm_l{}'.format(i+2))(nodes_encs)
        self.fill_lextree_encs_sage(prefixtree, nodes_encs)

    def forward_sage(self, embeddings, adjacency, layerid=1):
        if layerid == 1:
            nodes_encs = torch.relu(getattr(self, 'sage_pool_{}'.format(layerid))(embeddings))
        else:
            nodes_encs = embeddings
        pooled_encs = torch.einsum('ij,jk->ik', adjacency, nodes_encs)

        pooled_encs = torch.cat([embeddings, pooled_encs], dim=1)
        pooled_encs = torch.relu(getattr(self, 'sage_merge_{}'.format(layerid))(pooled_encs))
        return pooled_encs


class APPNP(torch.nn.Module):
    def __init__(self, embdim, treehid, nlayer, dropout, residual=False, maxpool=False):
        super(APPNP, self).__init__()
        self.treehid = treehid
        self.embdim = embdim
        self.ff_1 = torch.nn.Linear(self.embdim, self.treehid)
        self.dropout = torch.nn.Dropout(dropout)
        self.nlayer = nlayer
        self.residual = residual
        self.alpha = 0.2
        self.maxpool = maxpool

    def get_lextree_encs(self, decemb, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = len(embeddings)
            ey = decemb.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[1] == -1 and lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                ey = decemb.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs(decemb, values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def fill_lextree_encs(self, lextree, nodes_encs, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[1] == -1 and lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def maxpool_fn(self, nodes_encs, adjacency):
        pooled_encs = [0] * len(adjacency)
        for i, node in enumerate(adjacency):
            if len(node) > 1:
                candidates = nodes_encs[node]
                pooled_encs[node[0]] = candidates.max(dim=0)[0]
            elif len(node) == 1:
                pooled_encs[node[0]] = nodes_encs.new_zeros(nodes_encs.size(1))
        pooled_encs = torch.stack(pooled_encs)
        return pooled_encs

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs(decemb, prefixtree, embeddings, adjacency)
        n_nodes = len(embeddings)
        nodes_encs = torch.cat(embeddings, dim=0)
        # Calculate adjacency matrix
        adjacency_mat = nodes_encs.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)

        h_0 = torch.relu(self.ff_1(nodes_encs))
        h_i = h_0
        for i in range(self.nlayer):
            h_i = self.dropout(h_i)
            if self.maxpool:
                h_i = self.maxpool_fn(h_i, adjacency)
            else:
                h_i = torch.einsum('ij,jk->ik', adjacency_mat, h_i)
            h_i = (1 - self.alpha) * h_i + self.alpha * h_0
        self.fill_lextree_encs(prefixtree, h_i)


class GCNII(torch.nn.Module):
    def __init__(self, embdim, treehid, nlayer, dropout, residual=False, alpha=0.5, nhead=1, tied=False):
        super(GCNII, self).__init__()
        self.treehid = treehid
        self.embdim = embdim
        self.ff_1 = torch.nn.Linear(self.embdim, self.treehid)
        self.tied = tied
        for i in range(nlayer-1):
            setattr(self, 'ff_{}'.format(i+2), torch.nn.Linear(self.treehid, self.treehid))
            if self.tied:
                assert self.embdim == self.treehid
                getattr(self, 'ff_{}'.format(i+2)).weight = self.ff_1.weight
                getattr(self, 'ff_{}'.format(i+2)).bias = self.ff_1.bias
        self.dropout = torch.nn.Dropout(dropout)
        self.nlayer = nlayer
        self.residual = residual
        self.alpha = alpha
        self.nhead = nhead

    def get_lextree_encs(self, decemb, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = len(embeddings)
            ey = decemb.weight[wordpiece].unsqueeze(0)
            embeddings.append(self.dropout(ey))
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[1] == -1 and lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                ey = decemb.weight[wordpiece].unsqueeze(0)
                embeddings.append(self.dropout(ey))
            for newpiece, values in lextree[0].items():
                ids.append(self.get_lextree_encs(decemb, values, embeddings, adjacency, newpiece))
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def fill_lextree_encs(self, lextree, nodes_encs, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[1] == -1 and lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs(decemb, prefixtree, embeddings, adjacency)
        n_nodes = len(embeddings)
        nodes_encs = torch.cat(embeddings, dim=0)
        # Calculate adjacency matrix
        adjacency_mat = nodes_encs.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        adjacency_mat = torch.einsum('ij,jk->ik', degrees, adjacency_mat)
        adjacency_mat = torch.einsum('ij,jk->ik', adjacency_mat, degrees)

        h_0 = torch.relu(self.ff_1(nodes_encs))
        h_i = h_0
        all_node_encs = [h_0]
        for i in range(self.nlayer-1):
            h_i = self.dropout(h_i)
            h_i = torch.einsum('ij,jk->ik', adjacency_mat, h_i) # PH
            h_i = (1 - self.alpha) * h_i + self.alpha * h_0 # (1-alpha) * PH + alpha * H0
            beta = math.log10(1.0 + 1.0 / (i + 1))
            h_i_proj = getattr(self, 'ff_{}'.format(i+2))(h_i)
            h_i = (1 - beta) * h_i + beta * h_i_proj # (1 - beta) * I + beta * W
            h_i = torch.relu(h_i)
            all_node_encs.append(h_i)
        if self.nhead > 1:
            output_encs = all_node_encs[0:1] + all_node_encs[-self.nhead+1:]
            h_i = torch.cat(output_encs, dim=-1)
        self.fill_lextree_encs(prefixtree, h_i)

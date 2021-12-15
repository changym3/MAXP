import os
import glob
import json
import pickle
import argparse
import datetime as dt
from pytorch_lightning.core import datamodule
import tqdm
import warnings
from easydict import EasyDict
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics import Accuracy

import dgl

import pytorch_lightning as pl


class GraphDataset:
    def __init__(self, base_path, etypes, degrees=[-1]):
        self.etypes = etypes
        self.graph, self.feats, self.labels = self.load_graph(base_path, etypes)
        self.default_train_nid, self.default_val_nid, self.default_predict_nid = self.load_default_split(base_path)
        self.train_nid = self.filter_out_degrees(self.default_train_nid, degrees)
        self.val_nid = self.filter_out_degrees(self.default_val_nid, degrees)
        self.predict_nid = self.filter_out_degrees(self.default_predict_nid, degrees)

    def load_graph(self, base_path, etypes):
        data_dict = {}
        df_path = os.path.join(base_path, 'edge_df.feather')
        cite_df = pd.read_feather(df_path)
        if 'cite' in etypes:
            data_dict[('paper', 'cite', 'paper')] = (cite_df.src_nid.values, cite_df.dst_nid.values)
        if 'cited' in etypes:
            data_dict[('paper', 'cited', 'paper')] = (cite_df.dst_nid.values, cite_df.src_nid.values)
        if 'bicited' in etypes:
            srcs = np.concatenate((cite_df.src_nid.values, cite_df.dst_nid.values))
            dsts = np.concatenate((cite_df.dst_nid.values, cite_df.src_nid.values))
            data_dict[('paper', 'bicited', 'paper')] = (srcs, dsts)
        graph = dgl.heterograph(data_dict)

        with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
        labels = torch.from_numpy(label_data['label']).long()
        features = np.load(os.path.join(base_path, 'features.npy'))
        node_feat = torch.from_numpy(features)
        graph.create_formats_()
        return graph, node_feat, labels

    def load_default_split(self, base_path):
        with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
        tr_label_idx = label_data['tr_label_idx']
        val_label_idx = label_data['val_label_idx']
        test_label_idx = label_data['test_label_idx']
        tr_label_idx = torch.from_numpy(tr_label_idx).long()
        val_label_idx = torch.from_numpy(val_label_idx).long()
        test_label_idx = torch.from_numpy(test_label_idx).long()
        return tr_label_idx, val_label_idx, test_label_idx

    def filter_out_degrees(self, nid, degrees):
        if degrees == [-1]:
            return nid
        
        degree_idxs = []
        for deg in degrees:
            idx,  = (self.graph.in_degrees(nid)==deg).nonzero(as_tuple=True)
            degree_idxs.append(idx)
        degree_nid = nid[torch.cat(degree_idxs, dim=-1)]
        return degree_nid

    def add_self_loop(self):
        for et in self.etypes:
            self.graph = dgl.add_self_loop(self.graph, etype=et)

    def to(self, device):
        self.graph, self.feats, self.labels = self.graph.to(device), self.feats.to(device), self.labels.to(device)
        self.train_nid, self.val_nid, self.predict_nid = self.train_nid.to(device), self.val_nid.to(device), self.predict_nid.to(device)
        return self

    def setup(self):
        self.graph.ndata['features'] = self.feats
        self.graph.ndata['labels'] = self.labels


class NodeIDDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.dataset.train_nid, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=2)
        return loader
    
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.dataset.val_nid, batch_size=self.config.batch_size, drop_last=False, num_workers=2)
        return loader

    def predict_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.dataset.predict_nid, batch_size=self.config.batch_size, drop_last=False, num_workers=2)
        return loader

    # def train_val_dataloader(self):
    #     train_val_nid = torch.cat([self.dataset.train_nid, self.dataset.val_nid], dim=-1)
    #     loader = torch.utils.data.DataLoader(
    #         train_val_nid, batch_size=self.config.batch_size, drop_last=False, num_workers=2)
    #     return loader

    # def train_val_predict_dataloader(self):
    #     train_val_predict_nid = torch.cat([self.dataset.train_nid, self.dataset.val_nid, self.dataset.predict_nid], dim=-1)
    #     loader = torch.utils.data.DataLoader(
    #         train_val_predict_nid, batch_size=self.config.batch_size, drop_last=False, num_workers=2)
    #     return loader
    
    def all_nid_dataloader(self):
        loader = torch.utils.data.DataLoader(
            torch.arange(self.dataset.graph.num_nodes()), batch_size=self.config.batch_size, drop_last=False, num_workers=0)
        return loader

        
class GraphDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset, device):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device

        self.dataset.to(device)
        self.dataset.setup()

    def get_sample_loader(self, nid):
        loader = dgl.dataloading.NodeDataLoader(self.dataset.graph, nid, 
                                                dgl.dataloading.MultiLayerNeighborSampler(self.config['fanouts']),
                                                device=self.device,
                                                batch_size=self.config['batch_size'],
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=self.config['num_workers']
                                                )
        loader.set_epoch = None
        return loader

    def get_inference_loader(self, nid):
        if self.config['num_layers'] == 1:
            fanouts = [50]
        elif self.config['num_layers'] == 2:
            fanouts = [30, 50]
        elif self.config['num_layers'] == 3:
            fanouts = [15, 30, 50]
        loader = dgl.dataloading.NodeDataLoader(self.dataset.graph, nid, 
                                                dgl.dataloading.MultiLayerNeighborSampler(fanouts),
                                                device=self.device,
                                                batch_size=self.config['batch_size'] // 4,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=self.config['num_workers']
                                                )
        loader.set_epoch = None
        return loader

    def train_dataloader(self):
        return self.get_sample_loader(self.dataset.train_nid)
    
    def val_dataloader(self):
        return self.get_sample_loader(self.dataset.val_nid)

    def predict_dataloader(self):
        return self.get_inference_loader(self.dataset.predict_nid)

    def all_nid_dataloader(self):
        all_nid = torch.arange(self.dataset.graph.num_nodes()).to(self.device)
        return self.get_inference_loader(all_nid)



class Prediction:
    def __init__(self, nids, preds, labels) -> None:
        self.num_nodes = 3655452
        self.num_classes = 23
        self.predictions = torch.zeros(self.num_nodes, self.num_classes)
        self.labels = torch.zeros(self.num_nodes).long()
        self.predictions[nids] = preds
        self.labels[nids] = labels

    def load_from_path(path):
        with open(path, 'rb') as f:
            teacher = pickle.load(f)
        #     nids, preds, labels = data
        # teacher = Prediction(nids, preds, labels)
        return teacher

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def to(self, device):
        self.predictions = self.predictions.to(device)
        self.labels = self.labels.to(device)
        return self
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
from srcs.GNN.gnn_lightning import MaxpLightning
from srcs.utils.utils import graph_predict_with_labels



class NodeFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feats = self.feats[idx]
        labels = self.labels[idx]
        return feats, labels


class ModelPredictor():
    def __init__(self, model, datamodule, device):
        self.model = model
        self.datamodule = datamodule
        self.device = device

    def predict(self, nids, use_inference_loader=True):
        if isinstance(self.model, MaxpLightning):
            if use_inference_loader:
                loader = self.datamodule.get_inference_loader(nids)
            else:
                loader = self.datamodule.get_sample_loader(nids)
            nids, preds, labels = self._graph_predict(loader)
        return nids, preds, labels

    @torch.no_grad()
    def _graph_predict(self, loader):
        model = self.model.to(self.device)
        model.eval()
        nodes = []
        preds = []
        labels = []
        with tqdm.tqdm(loader) as tq:
            for batch in tq:
                _, output_nodes, mfgs = batch
                batch_pred = model(batch)
                batch_labels = mfgs[-1].dstdata['labels']
                nodes.append(output_nodes)
                preds.append(batch_pred)
                labels.append(batch_labels)
        nodes = torch.cat(nodes, dim=0).cpu()
        preds = torch.cat(preds, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        return nodes, preds, labels



        

# class Ensembler():
#     def __init__(self, ensemble_type):
#         self.ensemble_type = ensemble_type

#     def init_data(self, res_list):
#         self.res_list = res_list
#         pred_df_list = []
#         for (nids, preds, labels) in self.res_list:
#             preds = scipy.special.softmax(preds, axis=1)
#             pred_df = pd.DataFrame(preds, index=nids).sort_index()
#             pred_df_list.append(pred_df)
#         self.pred_df_list = pred_df_list
#         self.label_df = pd.DataFrame(labels, index=nids, columns=['labels']).sort_index()
        

#     def fit(self, train_nid):
#         if self.ensemble_type == 'avg':
#             pass


#     def predict(self, predict_nid):
#         if self.ensemble_type == 'avg':
#             res_df = 0
#             for pred_df in self.pred_df_list:
#                 res_df += pred_df.loc[predict_nid]
#             nids = res_df.index 
#             preds = res_df.values
#             return nids, preds

#     def evaluate(self, test_nid):
#         nids, preds = self.predict(test_nid)
#         ensemble_result_log = {
#             'ensemble': self._evaluate_acc(nids, preds)
#         }
#         for i, (nids, preds, _) in enumerate(self.res_list):
#             ensemble_result_log[f'fold_{i}'] = self._evaluate_acc(nids, preds)
#         return ensemble_result_log
    
#     def _evaluate_acc(self, nids, preds):
#         labels = self.label_df.loc[nids, 'labels'].values
#         return accuracy_score(labels, preds.argmax(axis=1))
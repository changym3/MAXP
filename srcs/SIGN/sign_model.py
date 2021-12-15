import gc
import os
import pickle

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from easydict import EasyDict
from srcs.utils.nn import ChannelCombine, get_optimizer
from torchmetrics import Accuracy


def feat_average(dataset, num_layers, etypes):
    feats = []
    for et in etypes:
        g = dataset.graph.edge_type_subgraph([et])
        feat = dataset.feats
        feat_et = graph_feat_average(g, feat, num_layers) # obtain L+1 channels, N*(L+1)*d
        feats.append(feat_et)
        del feat_et
        gc.collect()
    feats = torch.cat(feats, dim=-1) # N*(L+1)* (E*d)
    return feats


def graph_feat_average(g, feat, num_layers):
    with g.local_scope():
        g.ndata["feat_0"] = feat
        print('Obtain {}-hop features'.format(0))
        for hop in range(1, num_layers + 1):
            g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"), fn.mean("msg", f"feat_{hop}"))
            print('Obtain {}-hop features'.format(hop))
        feats = []
        for hop in range(num_layers + 1):
            feats.append(g.ndata.pop(f"feat_{hop}"))
        feats = torch.stack(feats, dim=1) # N * (L+1) * d
        # feats = torch.stack(res, dim=0) # (L+1) * N * d
    return feats


class SIGN(pl.LightningModule):
    def __init__(self, config, dataset, device, steps_per_epoch):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.feats = dataset.feats
        self.labels = dataset.labels
        self.device_id = device
        self.steps_per_epoch = steps_per_epoch
        
        self.num_channels = config['num_layers'] + 1
        num_features = config['num_features']
        num_hiddens = config['num_hiddens']
        num_classes = config['num_classes']
        num_etypes = len(config['etypes'])

        self.Ws = nn.ModuleList()
        for i in range(self.num_channels):
            self.Ws.append(nn.Linear(num_features*num_etypes, num_hiddens))
        self.layer_combine = ChannelCombine(num_hiddens, self.config['layer_combine'], self.num_channels)
        self.MLPs = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_classes)
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def classification(self, h, h_init):
        return self.MLPs(h)

    def forward(self, X):
        X = X.permute(1, 0, 2) # to C*N*d
        h_list = []
        for i in range(self.num_channels):
            h_list.append(self.Ws[i](X[i]))
        h = torch.stack(h_list, 0)
        h = self.layer_combine(h)
        h = self.classification(h, X[0])
        return h
    
    def training_step(self, batch, batch_idx):
        nid = batch
        X = self.feats[nid].to(self.device_id)
        y = self.labels[nid].to(self.device_id)
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.train_acc(torch.softmax(logits, 1), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        nid = batch
        X = self.feats[nid].to(self.device_id)
        y = self.labels[nid].to(self.device_id)
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.val_acc(torch.softmax(logits, 1), y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        self.log("hp/loss", loss)
        self.log("hp/acc", self.val_acc)
        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/acc": 0, "hp/loss": 0})
        
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        return get_optimizer(self.config, self.parameters(), self.steps_per_epoch)
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        return [model_checkpoint]

    @torch.no_grad()
    def predict(self, loader, device):
        model = self.to(device)
        model.eval()

        all_nids= []
        all_preds = []
        all_labels = []
        with tqdm.tqdm(loader) as tq:
            for nids in tq:
                X = self.feats[nids].to(device)
                y = self.labels[nids].to(device)
                preds = model(X)
                all_nids.append(nids)
                all_preds.append(preds)
                all_labels.append(y)
        all_nids = torch.cat(all_nids, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_nids, all_preds, all_labels = all_nids.cpu(), all_preds.cpu(), all_labels.cpu()
        return all_nids, all_preds, all_labels
    
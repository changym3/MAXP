import os
import pickle
from torch.functional import split
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import pytorch_lightning as pl
from torchmetrics import Accuracy

from .gnn_model import HetGraphModel
from ..utils_nn import get_optimizer





class MaxpLightning(pl.LightningModule):
    def __init__(self, config, steps_per_epoch=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.steps_per_epoch = steps_per_epoch
        
        num_hiddens = config['num_hiddens']
        num_classes = config['num_classes']
        self.gnn = self.configure_gnn()
        self.classification = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_classes)
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
    def forward(self, batch):
        input_nodes, output_nodes, mfgs = batch
        batch_emb = self.gnn(mfgs)
        batch_pred = self.classification(batch_emb)
        return batch_pred
    
    def training_step(self, batch, batch_idx):
        batch_pred = self(batch)
        _, _, mfgs = batch
        batch_labels = mfgs[-1].dstdata['labels']
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.train_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_pred = self(batch)
        _, _, mfgs = batch
        batch_labels = mfgs[-1].dstdata['labels']
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("hp/loss", loss)
        self.log("hp/acc", self.val_acc)
        return loss
    
    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/acc": 0, "hp/loss": 0})
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        return get_optimizer(self.config, self.parameters(), self.steps_per_epoch)
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1)
        return [model_checkpoint]
    
    def configure_gnn(self):
        model = HetGraphModel(self.config)
        return model


class MaxpDataModule(pl.LightningDataModule):
    def __init__(self, config, graph_data, split_idx, device):
        super().__init__()
        self.config = config
        self.device = device
        graph, node_feat, labels = graph_data
        graph.ndata['features'] = node_feat
        graph.ndata['labels'] = labels
        self.graph = graph.to(device)
        train_nid, val_nid, predict_nid = split_idx
        self.train_nid, self.val_nid, self.predict_nid = train_nid.to(self.device), val_nid.to(self.device), predict_nid.to(self.device)

    def get_sample_loader(self, nid):
        nid = nid.to(self.device)
        loader = dgl.dataloading.NodeDataLoader(self.graph, nid, 
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
        nid = nid.to(self.device)
        if self.config['num_layers'] == 1:
            fanouts = [50]
        elif self.config['num_layers'] == 2:
            fanouts = [30, 50]
        elif self.config['num_layers'] == 3:
            fanouts = [15, 30, 50]
        loader = dgl.dataloading.NodeDataLoader(self.graph, nid, 
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
        return self.get_sample_loader(self.train_nid)
    
    def val_dataloader(self):
        return self.get_sample_loader(self.val_nid)

    def predict_dataloader(self):
        return self.get_full_loader(self.predict_nid)
    
    def describe(self):
        print('################ Graph info: ###############')
        print(self.graph)
        print('################ Label info: ################')
        print('Total labels (including not labeled): {}'.format(self.graph.ndata['labels'].shape[0]))
        print('               Training label number: {}'.format(self.train_nid.shape[0]))
        print('             Validation label number: {}'.format(self.val_nid.shape[0]))
        print('                   Test label number: {}'.format(self.predict_nid.shape[0]))
        print('################ Feature info: ###############')
        print('Node\'s feature shape:{}'.format(self.graph.ndata['features'].shape))


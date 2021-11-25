import os
import json
import pickle
import argparse
import datetime as dt
import tqdm
import warnings
from easydict import EasyDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics import Accuracy

import dgl
import dgl.multiprocessing as mp

import srcs.graph_model as gm


class MaxpLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
            
        self.gnn = self.configure_gnn()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
    def forward(self, batch):
        input_nodes, output_nodes, mfgs = batch
        batch_inputs = mfgs[0].srcdata['features']
        batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.gnn(mfgs, batch_inputs)
        return batch_pred
    
    def training_step(self, batch, batch_idx):
        batch_pred = self(batch)
        _, _, mfgs = batch
        batch_labels = mfgs[-1].dstdata['labels']
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.train_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_pred = self(batch)
        _, _, mfgs = batch
        batch_labels = mfgs[-1].dstdata['labels']
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log("hp/loss", loss)
        self.log("hp/acc", self.val_acc)
        return loss
    
    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/acc": 0, "hp/loss": 0})
        
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
    #     scheduler = {
    #         'scheduler': OneCycleLR(optimizer,  max_lr=self.learning_rate, steps_per_epoch=self.steps_per_epoch, epochs=1, div_factor=2, final_div_factor=1.2, verbose=False),
    #         'interval': 'step',
    #         'frequency': 1
    #     }
    #     return [optimizer], [scheduler]
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer
    
    def configure_callbacks(self):
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=self.config.patience)
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        return [early_stopping, model_checkpoint]
    
        
    def configure_gnn(self):
        num_feats = self.config['NUM_FEATS']
        num_classes = self.config['NUM_CLASSES']
        hidden_dim = self.config['hidden_dim']
        num_layers = self.config['num_layers']
        gnn_model = self.config['gnn_model']
        
        num_heads = self.config['num_heads']
        feat_drop = self.config['feat_drop']
        attn_drop = self.config['attn_drop']
        
        
        if gnn_model == 'graphsage':
            model = gm.GraphSageModel(num_feats, hidden_dim, num_layers, num_classes)
        elif gnn_model == 'graphconv':
            model = gm.GraphConvModel(num_feats, hidden_dim, num_layers, num_classes,
                                   norm='both', activation=F.relu, dropout=0)
        elif gnn_model == 'graphattn':
            model = gm.GraphAttnModel(num_feats, hidden_dim, num_layers, num_classes,
                                   heads=num_heads, activation=F.relu, feat_drop=feat_drop, attn_drop=attn_drop)
        elif 'Het' in gnn_model:
            model = gm.HetGraphModel(self.config)
        else:
            raise NotImplementedError('Not Implemented')
        return model

    @torch.no_grad()
    def graph_inference(self, model, graph, nid, device):
        config = self.config
        sampler = dgl.dataloading.MultiLayerNeighborSampler(config['fanout'])
        loader = dgl.dataloading.NodeDataLoader(graph, nid, sampler,
                                                device=device,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=config['num_workers']
                                                )
        loader.set_epoch = None
        model = model.to(device)
        model.eval()
        
        nodes = []
        preds = []
        with tqdm.tqdm(loader) as tq:
            for step, batch in enumerate(tq):
                _, output_nodes, _ = batch
                batch_pred = model(batch)
                nodes.append(output_nodes)
                preds.append(batch_pred)
            # tq.set_postfix({'step': '%.03f' % loss.item()}, refresh=False)
            tq.set_description("Inference:")
        nodes = torch.cat(nodes, dim=0).cpu()
        preds = torch.cat(preds, dim=0).cpu()

        model.train()
        return nodes, preds
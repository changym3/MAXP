import os
import glob
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
import dgl.function as fn


from srcs.GNN.gnn_lightning import HetGraphModel
from srcs.model_utils import get_optimizer


class RandomSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, g):
        super().__init__(len(fanouts))
        self.fanouts = fanouts
        self.weights = g.in_degrees().float() ** 0.75

    def sample_frontier(self, block_id, g, seed_nodes):
        if isinstance(seed_nodes, dict):
            assert len(seed_nodes) == 1, "len(g.ntypes) != 1"
            for k in seed_nodes:
                seed_nodes = seed_nodes[k]
        # Get all inbound edges to `seed_nodes`
        fanout = self.fanouts[block_id]
        src = self.weights.multinomial(num_samples = len(seed_nodes)*fanout, replacement=True)
        dst = seed_nodes.repeat_interleave(fanout)
        frontier = dgl.heterograph(data_dict={('paper','bicited', 'paper') : (src, dst)},
                                   num_nodes_dict={'paper': g.number_of_nodes()})
        # frontier = dgl.graph((src, dst), num_nodes=g.num_nodes())
        # frontier.ndata['features'] = g.ndata['features']
        # frontier.ndata['labels'] = g.ndata['labels']
        return frontier

    def __len__(self):
        return len(self.fanouts)



class MaxpLightningWithRandomSampler(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.gnn = self.configure_gnn()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def complete_blocks(self, mfgs):
        g = self.trainer.datamodule.graph
        blocks = []
        for block in mfgs:
            src_idx = block.srcdata[dgl.NID]
            dst_idx = block.dstdata[dgl.NID]
            block.srcdata['features'] = g.ndata['features'][src_idx]
            block.srcdata['labels'] = g.ndata['labels'][src_idx]
            block.dstdata['features'] = g.ndata['features'][dst_idx]
            block.dstdata['labels'] = g.ndata['labels'][dst_idx]
            blocks.append(block)
        return blocks
        
    def forward(self, batch):
        input_nodes, output_nodes, mfgs = batch
        mfgs = self.complete_blocks(mfgs)
        # batch_inputs = mfgs[0].srcdata['features']
        # batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.gnn(mfgs)
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
        return get_optimizer(self.config, self.parameters(), self.trainer.datamodule.train_dataloader())
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        return [model_checkpoint]
    
    def configure_gnn(self):
        model = HetGraphModel(self.config)
        return model
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

from srcs.SIGN.sign_model import SIGNDataModule
from srcs.GNN.gnn_lightning import HetGraphModel
from srcs.utils.nn import NodeFeatureDataset, get_optimizer
from srcs.utils.utils import load_graph, get_trainer


class ReconstructionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        num_features = config['num_features']
        num_hiddens = config['num_hiddens']
        self.dist = '2-norm'

        self.MLPs = nn.Sequential(
            nn.Linear(num_features, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens)
        )
        if self.dist == 'MLP':
            self.mlp_dist_metrics = nn.Sequential(
                nn.Linear(num_hiddens * 2, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
                nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
                nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
                nn.Linear(num_hiddens, 1), nn.Sigmoid()
            )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, X):
        h = self.MLPs(X)
        return h
    
    def distance(self, X, X_prime):
        if self.dist == '2-norm':
            dist = torch.linalg.vector_norm(X-X_prime, ord=2, dim=-1, keepdim=True).mean()
        elif self.dist == 'cos':
            dist = F.cosine_similarity(X, X_prime, dim=-1).mean()
        elif self.dist == 'MLP':
            input = torch.cat([X, X_prime], dim=-1)
            dist = 1 - self.mlp_dist_metrics(input)
        return dist
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        new_X = self(X)
        loss = self.distance(new_X, y)
        # self.train_acc(torch.softmax(new_X, 1), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        # self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.distance(logits, y)
        self.val_acc(torch.softmax(logits, 1), y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        # self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        self.log("hp/loss", loss)
        # self.log("hp/acc", self.val_acc)
        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/acc": 0, "hp/loss": 0})
        
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        return get_optimizer(self.config, self.parameters(), self.trainer.datamodule.train_dataloader())
    
    # def configure_callbacks(self):
        # model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        # return [model_checkpoint]

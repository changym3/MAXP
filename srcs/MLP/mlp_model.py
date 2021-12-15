import os
import gc
import pickle
from easydict import EasyDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy
import tqdm

from srcs.utils.nn import get_optimizer


class MLPModel(pl.LightningModule):
    def __init__(self, config, dataset, device, steps_per_epoch):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.feats = dataset.feats
        self.labels = dataset.labels
        self.device_id = device
        self.steps_per_epoch = steps_per_epoch

        num_features = config['num_features']
        num_hiddens = config['num_hiddens']
        num_classes = config['num_classes']

        self.MLPs = nn.Sequential(
            nn.Linear(num_features, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_classes)
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, X):
        h = self.MLPs(X)
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
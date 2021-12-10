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

from srcs.SIGN.sign_nn import SIGNDataModule
from srcs.GNN.gnn_lightning import HetGraphModel
from srcs.model_utils import NodeFeatureDataset, get_optimizer, get_trainer
from srcs.utils import load_graph



class OnlyDegreeDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset, only_degrees):
        super().__init__()
        self.config = config
        self.graph, self.feats, self.labels = dataset
        self.only_degrees = only_degrees
        self.raw_train_nid, self.raw_val_nid, self.raw_test_nid = self.split_labels(config['data_dir'])
        self.train_nid = self.generate_degree_nid(self.raw_train_nid)
        self.val_nid = self.generate_degree_nid(self.raw_val_nid)
        self.test_nid = self.generate_degree_nid(self.raw_test_nid)

    def generate_degree_nid(self, nid):
        res = []
        for degree in self.only_degrees:
            degree_idx = (self.graph.in_degrees(nid)==degree).nonzero(as_tuple=True)
            res.append(degree_idx)
        degree_idx = torch.cat(res, dim=-1)
        degree_nid = nid[degree_idx]
        return degree_nid

    def generate_dataloader(self, nid):
        dataset = NodeFeatureDataset(self.feats[nid], self.labels[nid])
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=self.config['num_workers'])
        return loader

    def train_dataloader(self):
        return self.generate_dataloader(self.train_nid)

    def val_dataloader(self):
        return self.generate_dataloader(self.val_nid)

    def split_labels(self, data_dir, splits='default'):
        with open(os.path.join(data_dir, 'labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
        tr_label_idx = label_data['tr_label_idx']
        val_label_idx = label_data['val_label_idx']
        test_label_idx = label_data['test_label_idx']
        tr_label_idx = torch.from_numpy(tr_label_idx).long()
        val_label_idx = torch.from_numpy(val_label_idx).long()
        test_label_idx = torch.from_numpy(test_label_idx).long()
        return tr_label_idx, val_label_idx, test_label_idx


class MLPModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

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
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        self.train_acc(torch.softmax(logits, 1), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
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
        return get_optimizer(self.config, self.parameters(), self.trainer.datamodule.train_dataloader())
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        return [model_checkpoint]



def get_parser():
    parser = argparse.ArgumentParser()
    # data-info
    parser.add_argument('--data_dir', type=str, default='/home/changym/competitions/MAXP/official_data/processed_data/')
    parser.add_argument('--sub_dir', type=str, default='/home/changym/competitions/MAXP/official_data/submissions/')
    parser.add_argument('--num_features', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=23)
    parser.add_argument('--etypes', nargs='+', type=str, default=['bicited'])
    # model_info
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hiddens', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clipping_norm', type=float, default=0.5)
    # optim-related
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_norm', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='1cycle', choices=['none', '1cycle'])
    # training setting
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--inference', action='store_true')

    return parser


def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    graph, feats, labels = load_graph(config.data_dir, config.etypes)
    device = torch.device('cuda:{}'.format(config.gpu_id))
    datamodule = OnlyDegreeDataModule(config, (graph, feats, labels), config.only_degrees)
    model = MLPModel(config)
    trainer = get_trainer(config)
    trainer.fit(model, datamodule=datamodule)
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))
    print('The model is saved at', trainer.checkpoint_callback.best_model_path)
    print('The model performance of last epoch :', json.dumps(trainer.progress_bar_dict, indent=4))



if __name__ == '__main__':

    parser = get_parser()
    # graph specific params
    parser.add_argument('--only_degrees', type=int, nargs='+', default=[0])
    # args = parser.parse_args(''.split())
    
    args = parser.parse_args()
    config = EasyDict(vars(args))
    
    main(config)

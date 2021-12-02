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

import dgl
import dgl.function as fn

from ..model_utils import get_optimizer, ChannelCombine


def feat_average(g, feat, num_layer):
    with g.local_scope():
        g.ndata["feat_0"] = feat
        print('Obtain {}-hop features'.format(0))
        for hop in range(1, num_layer + 1):
            g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"), fn.mean("msg", f"feat_{hop}"))
            print('Obtain {}-hop features'.format(hop))
        feats = []
        for hop in range(num_layer + 1):
            feats.append(g.ndata.pop(f"feat_{hop}"))
        feats = torch.stack(feats, dim=1) # N * (L+1) * d
        # feats = torch.stack(res, dim=0) # (L+1) * N * d
    return feats


class SIGNDataset(torch.utils.data.Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        feats = self.feats[idx]
        labels = self.labels[idx]
        return feats, labels

class SIGNDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.graph, self.feats, self.labels = dataset
        self.train_nid, self.val_nid, self.test_nid = self.split_labels(config['data_dir'])

    def train_dataloader(self):
        dataset = SIGNDataset(self.feats[self.train_nid], self.labels[self.train_nid])
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=self.config['num_workers'])
        return loader
    
    def val_dataloader(self):
        dataset = SIGNDataset(self.feats[self.val_nid], self.labels[self.val_nid])
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=self.config['num_workers'])
        return loader
    
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
    
    
class SIGN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        self.Ws = nn.ModuleList()
        self.num_channels = config['num_layers'] + 1
        num_features = config['num_features']
        num_hiddens = config['num_hiddens']
        num_classes = config['num_classes']
        num_etypes = len(config['etypes'])
        for i in range(self.num_channels):
            self.Ws.append(nn.Linear(num_features*num_etypes, num_hiddens))
        
        # self.combine_transform = nn.Sequential(
        #     nn.Linear(self.num_channels * num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout'])
        # )
        # self.combine_attention = nn.Sequential(
        #     nn.Linear(num_hiddens, num_hiddens), nn.Tanh(), nn.Linear(num_hiddens, 1)
        # )
        # self.combine_attention_global = nn.Sequential(
        #     nn.Linear(num_hiddens*2, num_hiddens), nn.Tanh(), nn.Linear(num_hiddens, 1)
        # )
        self.layer_combine = ChannelCombine(num_hiddens, self.config['layer_combine'], self.num_channels)

        self.MLPs = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_hiddens), nn.BatchNorm1d(num_hiddens), nn.LeakyReLU(), nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_classes)
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    # def layer_combination(self, h_list):
    #     if self.config['layer_combination'] == 'transform':
    #         h = torch.cat(h_list, dim=-1)
    #         h = self.combine_transform(h)
    #     elif self.config['layer_combination'] == 'last':
    #         h = h_list[-1]
    #     elif self.config['layer_combination'] == 'mean':
    #         h = torch.stack(h_list, 0).mean(dim=0)
    #     elif self.config['layer_combination'] == 'simple_attn':
    #         h = torch.stack(h_list, 0)
    #         attn = self.combine_attention(h).softmax(dim=0)
    #         h = (h*attn).sum(dim=0)
    #     elif self.config['layer_combination'] == 'global_attn':
    #         h = torch.stack(h_list, 0) # L*N*d
    #         global_h = self.combine_transform(torch.cat(h_list, dim=-1)) # N*d
    #         global_h = global_h.repeat(self.num_channels, 1, 1) # L*N*d
    #         attn_input = torch.cat((h, global_h), dim=-1) # L*N*2d
    #         attn = self.combine_attention_global(attn_input).softmax(dim=0) # L*N*1
    #         h = (h*attn).sum(dim=0)
    #     return h

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
        return get_optimizer(self.config, self.parameters())
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        return [model_checkpoint]
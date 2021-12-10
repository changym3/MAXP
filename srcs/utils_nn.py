import argparse
from easydict import EasyDict
from pytorch_lightning import callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from srcs.utils import graph_predict_with_labels


def get_optimizer(config, parameters, steps_per_epoch):
    # elif (self.sched == 'cosine'):
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
    # elif (self.sched == 'cosinewr'):
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.max_epochs/4))
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)
    if config['optimizer'] == 'adam':
        optimizer = optim.AdamW(parameters, lr=config['lr'], amsgrad=True, weight_decay=config['l2_norm'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=config['lr'], weight_decay=config['l2_norm'])
    elif config['optimizer'] == 'sgdm':
        optimizer = optim.SGD(parameters, lr=config['lr'], momentum=0.9, weight_decay=config['l2_norm'])
    elif config['optimizer'] == 'sgdm-nag':
        optimizer = optim.SGD(parameters, lr=config['lr'], momentum=0.9, nesterov=True, weight_decay=config['l2_norm'])
    

    if config['scheduler'] == '1cycle':
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=steps_per_epoch, epochs=config['num_epochs'],
                div_factor=2, final_div_factor=10, verbose=False),
            'interval': 'step',
        }
    # elif config['scheduler'] == 'step':
    #     scheduler = {
    #         'scheduler': optim.lr_scheduler.stepLR(optimizer, step_size=5, gamma=0.5),
    #         'interval': 'epoch',
    #     }
    elif config['scheduler'] == 'plateau':
        scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.1, threshold_mode='abs', threshold=0.25), 
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_acc",
        }
    elif config['scheduler'] == 'cosine':
        scheduler = {
                "scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs']), # 所有epoches学习率都在下降，下降曲线cosine
                "interval": "epoch",
                "frequency": 1,
        }
    elif config['scheduler'] == 'cosinewr':
        scheduler = {
                "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2),
                "interval": "epoch",
                "frequency": 1,
        }
    elif config['scheduler'] == 'none':
        return [optimizer]
    else:
        raise Exception("Not implemented scheduler")
    return [optimizer], [scheduler]


class ChannelCombine(nn.Module):
    def __init__(self, num_hiddens, combine_type, num_channels=None):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.dim_query = num_hiddens
        self.combine_type = combine_type
        self.num_channels = num_channels

        if combine_type in ['mean', 'last']:
            pass
        elif combine_type == 'transform':
            self.transformation = nn.Linear(num_hiddens * num_channels, num_hiddens)
        elif combine_type == 'simple_attn':
            self.attention = nn.Sequential(
                nn.Linear(num_hiddens, num_hiddens), nn.Tanh(), nn.Linear(num_hiddens, 1, bias=False)
            )
        elif combine_type == 'global_attn':
            self.transformation = nn.Linear(num_hiddens * num_channels, num_hiddens)
            self.global_attention = nn.Sequential(
                nn.Linear(num_hiddens*2, num_hiddens), nn.Tanh(), nn.Linear(num_hiddens, 1, bias=False)
            )
    
    def forward(self, h):
        # h \in (C, N, d)
        if self.combine_type == 'mean':
            h = h.mean(dim=0)
        elif self.combine_type == 'transform':
            # (C, N, d) to (N, C, d) to (N, C*d)
            h = h.permute(1, 0, 2).flatten(start_dim=1)
            h = self.transformation(h)
        elif self.combine_type == 'simple_attn':
            attn = F.softmax(self.attention(h), dim=0)
            h = (attn * h).sum(dim=0)
        elif self.combine_type == 'global_attn':
            global_h = h.permute(1, 0, 2).flatten(start_dim=1)  # (N, C*d)
            global_h = self.transformation(global_h)  # (N, d)
            global_h = global_h.repeat(self.num_channels, 1, 1) # (C, N, d)
            local_global = torch.cat((h, global_h), dim=-1) # (C, N, 2d)
            attn = self.global_attention(local_global) # (C, N, 1)
            h = (h*attn).sum(dim=0)
        return h



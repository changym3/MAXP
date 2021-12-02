
import argparse
from easydict import EasyDict
from pytorch_lightning import callbacks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl


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
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--inference', action='store_true')


    # graph specific params
    parser.add_argument('--fanout', type=int, default=10)
    parser.add_argument('--gnn_model', type=str, default='GAT')
    parser.add_argument('--het_combine', type=str, default='transform')
    parser.add_argument('--layer_combine', type=str, default='last')
    parser.add_argument('--num_heads', type=float, default=4)
    parser.add_argument('--feat_drop', type=float, default=0)
    parser.add_argument('--attn_drop', type=float, default=0)
    return parser


def get_trainer(config):
    tr_args = EasyDict()
    logger = pl.loggers.TensorBoardLogger(
        save_dir='./logs/', name=config.name, version=config.version,
        default_hp_metric=False,
    )
    tr_args['callbacks'] = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        # pl.callbacks.StochasticWeightAveraging(annealing_epochs=3)
        # pl.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=config.patience)
    ]
    # tr_args['accumulate_grad_batches'] = 64
    # tr_args['gradient_clip_val'] = config.clipping_norm


    tr_args['logger'] = logger
    tr_args['gpus'] = [config.gpu_id]
    tr_args['max_epochs'] = config.num_epochs
    trainer = pl.Trainer(
        **tr_args
    )
    return trainer


def get_optimizer(config, parameters, loader=None):
    # elif (self.sched == 'cosine'):
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
    # elif (self.sched == 'cosinewr'):
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.max_epochs/4))
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50, verbose=True)

    optimizer = optim.AdamW(parameters, lr=config['lr'], amsgrad=True, weight_decay=config['l2_norm'])
    
    if config['scheduler'] == '1cycle':
        steps_per_epoch = len(loader)
        # n_train = 939963
        # div, mod = divmod(n_train, config['batch_size'])
        # steps_per_epoch = div + 1 if mod !=0 else div
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=steps_per_epoch, epochs=config['num_epochs'],
                div_factor=2, final_div_factor=10, verbose=False),
            'interval': 'step',
        }
    elif config['scheduler'] == 'step':
        scheduler = {
            'scheduler': optim.lr_scheduler.stepLR(optimizer, step_size=5, gamma=0.5),
            'interval': 'epoch',
        }
    elif config['scheduler'] == 'plateau':
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=1),
            'interval': 'epoch',
            'monitor': 'val_loss',
        }
    elif config['scheduler'] == 'none':
        return [optimizer]
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



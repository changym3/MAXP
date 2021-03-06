
import argparse
import glob
import os
import pickle
from easydict import EasyDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl

import pytorch_lightning as pl
from pytorch_lightning import callbacks
import tqdm



def get_parser():
    parser = argparse.ArgumentParser()
    # data-info
    parser.add_argument('--data_dir', type=str, default='/home/changym/competitions/MAXP/official_data/processed_data/')
    parser.add_argument('--sub_dir', type=str, default='/home/changym/competitions/MAXP/submissions/')
    parser.add_argument('--num_features', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=23)
    parser.add_argument('--etypes', nargs='+', type=str, default=['bicited'])
    parser.add_argument('--use_degrees', nargs='+', type=int, default=[-1])
    # model_info
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_hiddens', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clipping_norm', type=float, default=0.0)
    # optim-related
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_norm', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='1cycle')
    # training setting
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--training_mode', type=str, default='evaluate', choices=['evaluate', 'inference'])
    parser.add_argument('--save_submission', action='store_true')

    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--ensemble_type', type=str, default='avg', choices=['avg', 'linear_voting'])
    parser.add_argument('--random_state', type=int, default=42)

    # pl trainer settings
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)

    # graph specific params
    parser.add_argument('--fanouts', type=int, nargs='+', default=[10, 10, 10])
    parser.add_argument('--gnn_model', type=str, default='Transformer')
    parser.add_argument('--het_combine', type=str, default='transform')
    parser.add_argument('--layer_combine', type=str, default='transform')
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop', type=float, default=0.2)

    # debug settings
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--fast_dev_run', type=int, default=10)
    parser.add_argument('--overfit_batches', type=int, default=10)
    parser.add_argument('--track_grad_norm', type=int, default=-1)

    return parser


def get_trainer(config):
    tr_args = get_trainer_args(config)
    trainer = pl.Trainer(
        **tr_args
    )
    return trainer

def get_trainer_args(config):
    tr_args = {}
    logger = pl.loggers.TensorBoardLogger(
        save_dir='./logs/', name=config.name, version=config.version,
        default_hp_metric=False,
    )
    tr_args['callbacks'] = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        # pl.callbacks.DeviceStatsMonitor(),
        # pl.callbacks.StochasticWeightAveraging(annealing_epochs=3)
        # pl.callbacks.EarlyStopping(monitor="val_loss", mode="max", patience=config.patience)
    ]
    # tr_args['gradient_clip_val'] = config.clipping_norm
    # tr_args['track_grad_norm'] = config.track_grad_norm

    # tr_args['accumulate_grad_batches'] = {2: 2,
    # 5:4
    # }
    # config.accumulate_grad_batches

    if config.dev:
        tr_args['fast_dev_run'] = config.fast_dev_run
        # tr_args['overfit_batches'] = config.overfit_batches

    tr_args['log_every_n_steps'] = 1
    tr_args['logger'] = logger
    tr_args['gpus'] = [config.gpu_id]
    tr_args['max_epochs'] = config.num_epochs
    return tr_args



def load_model(ModelClass, model_dir):
    assert issubclass(ModelClass, pl.LightningModule)
    ckpt_path = glob.glob(os.path.join(model_dir, '*', '*.ckpt'))[0]
    model = ModelClass.load_from_checkpoint(ckpt_path)
    return model





@torch.no_grad()
def graph_predict(model, loader, device):
    nodes, preds, _ = graph_predict_with_labels()
    return nodes, preds


@torch.no_grad()
def graph_predict_with_labels(model, loader, device):
    # loader = dgl.dataloading.NodeDataLoader(graph, train_nid, 
    #                                         dgl.dataloading.MultiLayerNeighborSampler(config['fanouts'])
    #                                         device=device,
    #                                         batch_size=self.config['batch_size'],
    #                                         shuffle=True,
    #                                         drop_last=False,
    #                                         num_workers=self.config['num_workers']
    #                                         )
    model = model.to(device)
    model.eval()
    nodes = []
    preds = []
    labels = []
    with tqdm.tqdm(loader) as tq:
        for batch in tq:
            _, output_nodes, mfgs = batch
            batch_pred = model(batch)
            batch_labels = mfgs[-1].dstdata['labels']
            nodes.append(output_nodes)
            preds.append(batch_pred)
            labels.append(batch_labels)
    nodes = torch.cat(nodes, dim=0)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    return nodes, preds, labels


def save_submission(nodes, preds, filename, data_dir, sub_dir):
    '''
    nodes and preds -> Tensor
    '''
    nodes = nodes.cpu().numpy()
    preds = preds.cpu().numpy()
    label_array = np.array(list('ABCDEFGHIJKLMNOPQRSTUVW'))
    node_labels = label_array[preds.argmax(axis=1)]
    nodes_df = pd.read_csv(os.path.join(data_dir, 'IDandLabels.csv'), dtype={'Label':str})
    paper_array = nodes_df.paper_id.values
    node_ids = paper_array[nodes]

    test_submission = pd.DataFrame({'id': node_ids, 'label': node_labels})
    sub_file_name = os.path.join(sub_dir, '{}.csv'.format(filename))
    test_submission.to_csv(sub_file_name, index=False)
    print('Finish saving submission to {}'.format(sub_file_name))

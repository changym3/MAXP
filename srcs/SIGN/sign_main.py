import argparse
import datetime as dt
import gc
import json
import os
import pickle

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
from easydict import EasyDict
import sys
print(__name__)

from ..utils import load_graph, save_submission
from ..model_utils import get_trainer, get_parser
from .sign_nn import SIGN, SIGNDataModule, feat_average

@torch.no_grad()
def sign_predict(model, datamodule, device, batch_size, num_workers=0):
    model.eval()
    model = model.to(device)
    loader = torch.utils.data.DataLoader(
        datamodule.test_nid, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers
    )
    feats = datamodule.feats
    all_nids= []
    all_preds = []
    with tqdm.tqdm(loader) as tq:
        for nids in tq:
            X = feats[nids].to(device)
            preds = model(X)
            all_nids.append(nids)
            all_preds.append(preds)
    all_nids = torch.cat(all_nids, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_nids, all_preds = all_nids.cpu(), all_preds.cpu()
    return all_nids, all_preds



def load_sign_model(cpkt_path):
    return SIGN.load_from_checkpoint(cpkt_path)



def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    graph, node_feat, labels = load_graph(config.data_dir, config.etypes)
    feats = []
    for et in graph.etypes:
        g = graph.edge_type_subgraph([et])
        feat_et = feat_average(g, node_feat, config.num_layers) # obtain L+1 channels, N*(L+1)*d
        feats.append(feat_et)
        del feat_et
        gc.collect()
    feats = torch.cat(feats, dim=-1) # N*(L+1)* (E*d)

    device = torch.device('cuda:{}'.format(config.gpu_id))
    datamodule = SIGNDataModule(config, (graph, feats, labels))
    model = SIGN(config)
    trainer = get_trainer(config)
    trainer.fit(model, datamodule=datamodule)
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))
    print('The model is saved at', trainer.checkpoint_callback.best_model_path)
    print('The model performance of last epoch :', json.dumps(trainer.progress_bar_dict, indent=4))

    if config.inference:
        nids, preds = sign_predict(model, datamodule, device, config.batch_size, config.num_workers)
        save_submission(nids, preds, 
            filename='{}_{}'.format(config.name, config.version), 
            data_dir=config.data_dir, sub_dir=config.sub_dir
        )


if __name__ == '__main__':

    parser = get_parser()

    parser.add_argument('--layer_combine', type=str, default='transform')
    parser.add_argument('--use_skip', action='store_true')
    # args = parser.parse_args(''.split())
    
    args = parser.parse_args()
    config = EasyDict(vars(args))
    
    main(config)





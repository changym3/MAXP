
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

from srcs.utils.utils import get_trainer, get_parser
from srcs.utils.data import GraphDataset, NodeIDDataModule
from .mlp_model import MLPModel


def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    dataset = GraphDataset(config.data_dir, config.etypes, config.use_degrees)
    datamodule = NodeIDDataModule(config, dataset)
    device = torch.device('cuda:{}'.format(config.gpu_id))

    model = MLPModel(config, dataset, device, steps_per_epoch=len(datamodule.train_dataloader()))
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

    # args = parser.parse_args(''.split())
    
    args = parser.parse_args()
    config = EasyDict(vars(args))

    config.num_epochs = 100
    config.batch_size = 10240

    main(config)


# python -m srcs.MLP.mlp_main --gpu_id 0 --name MLP_v1 --version base
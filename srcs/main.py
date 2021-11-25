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

from srcs.lightning_model import MaxpLightning
import srcs.graph_model as gm
from srcs.configs import *
from srcs.datamodule import MaxpDataModule


warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.")



    



def get_trainer(config):
    # directory/series/version
    logger = pl.loggers.TensorBoardLogger(
        save_dir='./{}/'.format(config.directory), default_hp_metric=False,
        name=config.series,
        version=config.version,
    )
    
    # about 1600 step to converge when lr=0.005
    trainer = pl.Trainer(
        logger = logger, gpus=[config.gpu_id],
        max_epochs=config.num_epochs,
        # max_epochs=1, # for test
        
        callbacks=[],
        
        val_check_interval=0.2,
        check_val_every_n_epoch=1,
        log_every_n_steps=50

    )
    return trainer


def load_model(directory='log', series='circle', version='latest'):
    if version == 'latest':
        version_pattern = os.path.join('.', directory, series, 'version_*')
        dirs = glob.glob(version_pattern)
        if len(dirs) == 0:
            raise Exception("There is no latest model.")
        latest_version = max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])
        version = 'version_{}'.format(latest_version)

    ckpt_path_pattern = os.path.join('.', directory, series, version, 'checkpoints', '*.ckpt')
    ckpt_path = glob.glob(ckpt_path_pattern)[0]
    ckpt = MaxpLightning.load_from_checkpoint(ckpt_path)
    return ckpt



def save_submission(model, datamodule, device, name):
    nodes, preds = model.graph_inference(model, datamodule.graph, datamodule.test_nid, device)

    label_array = np.array(list('ABCDEFGHIJKLMNOPQRSTUVW'))
    node_labels = label_array[preds.argmax(dim=1).numpy()]

    nodes_df = pd.read_csv('./official_data/processed_data/IDandLabels.csv', dtype={'Label':str})
    paper_array = nodes_df.paper_id.values
    node_ids = paper_array[nodes]

    test_submission = pd.DataFrame({'id': node_ids, 'label': node_labels})
    sub_file_name = './submissions/{}.csv'.format(name)
    test_submission.to_csv(sub_file_name, index=False)
    print('Finish saving submission to {}'.format(sub_file_name))



if __name__ == '__main__':

    config = get_cubic_config(series='circle')

    device = torch.device('cuda:{}'.format(config.gpu_id))
    datamodule = MaxpDataModule(config, device)
    model = MaxpLightning(config)
    trainer = get_trainer(config)
    trainer.fit(model=model, datamodule=datamodule)

    model = load_model(series='circle', version='latest')
    save_submission()




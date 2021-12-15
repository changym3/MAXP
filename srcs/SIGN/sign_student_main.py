import argparse
import datetime as dt
import gc
import json
import os
import pickle
import sys

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
from easydict import EasyDict
from srcs.utils.data import GraphDataset, NodeIDDataModule
from srcs.utils.utils import get_parser, get_trainer, save_submission

from .sign_student import SIGN, feat_average


def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    dataset = GraphDataset(config.data_dir, config.etypes, config.use_degrees)
    datamodule = NodeIDDataModule(config, dataset)
    device = torch.device('cuda:{}'.format(config.gpu_id))

    feats = feat_average(dataset, config.num_layers, config.etypes)
    dataset.feats = feats
    model = SIGN(config, dataset, device, steps_per_epoch=len(datamodule.train_dataloader()))
    trainer = get_trainer(config)
    trainer.fit(model, datamodule=datamodule)
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))
    print('The model is saved at', trainer.checkpoint_callback.best_model_path)
    print('The model performance of last epoch :', json.dumps(trainer.progress_bar_dict, indent=4))

    if config.save_submission:
        nids, preds = model.predict(datamodule.predict_dataloader(), device)
        save_submission(nids, preds, 
            filename='{}_{}'.format(config.name, config.version), 
            data_dir=config.data_dir, sub_dir=config.sub_dir
        )


if __name__ == '__main__':

    parser = get_parser()

    parser.add_argument('--teacher_path', type=str, default='/home/changym/competitions/MAXP/logs/sign_v1/base_L4/prediction.pkl')
    parser.add_argument('--kd_alpha', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=1)


    # args = parser.parse_args(''.split())
    
    args = parser.parse_args()
    config = EasyDict(vars(args))

    config.num_epochs = 20
    
    main(config)




# python -m srcs.SIGN.sign_student_main --gpu_id 0 --kd_alpha 1.0 --temperature 1 --num_layers 2 --name sign_v1 --version kd_a_1_t_1
# python -m srcs.SIGN.sign_student_main --gpu_id 0  --temperature 1 --name sign_v1  --num_layers 2
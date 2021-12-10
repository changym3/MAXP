from abc import ABC, abstractmethod
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics import Accuracy

from srcs.utils import get_trainer_args, load_default_split, load_graph, get_parser
from srcs.GNN.gnn_lightning import MaxpDataModule, MaxpLightning


class KFoldTrainer(ABC):
    def __init__(self, training_mode, cv, random_state, default_split, labels):
        super().__init__()
        self.default_train_nid, self.default_val_nid, self.default_predict_nid = default_split
        self.labels = labels

        self.training_mode = training_mode
        self.k = cv
        self.random_state = random_state

    # @abstractmethod
    def train_model(self):
        raise NotImplementedError
        
    def generate_kfold_nid(self):
        fold_idx = 0
        if self.training_mode == 'inference':
            yield fold_idx, self.default_train_nid, self.default_val_nid
            fold_idx += 1
            nid = torch.cat((self.default_train_nid, self.default_val_nid), dim=-1)
        elif self.training_mode == 'evaluate':
            nid = self.default_train_nid

        nid = nid.cpu().numpy()
        labels = self.labels.cpu().numpy()[nid]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        for (nid_train_idx, nid_test_idx) in skf.split(nid, labels):
            train_nid = torch.from_numpy(nid[nid_train_idx])
            val_nid = torch.from_numpy(nid[nid_test_idx])
            yield fold_idx, train_nid, val_nid
            fold_idx += 1


class KFoldGNNTrainer(KFoldTrainer):
    def __init__(self, config, datamodule, base_model_calss):
        super().__init__(
            config.training_mode, config.cv, config.random_state, 
            (datamodule.train_nid, datamodule.val_nid, datamodule.predict_nid),
            datamodule.graph.ndata['labels']
        )
        self.config = config
        self.datamodule = datamodule
        self.BaseModelCass = base_model_calss

    def train_model(self, config, fold_idx, train_nid, val_nid=None):
        tr_args = get_trainer_args(config)
        tr_args['logger'] = pl.loggers.TensorBoardLogger(
                                save_dir=f'./logs/ensemble_{config.training_mode}', name=config.name, version=f'fold_{fold_idx}',
                                default_hp_metric=False,
                            )
        trainer = pl.Trainer(
            **tr_args
        )
        if val_nid is not None:
            train_loader = self.datamodule.get_sample_loader(train_nid)
            val_loader = self.datamodule.get_sample_loader(val_nid)
            model = self.BaseModelCass(config, steps_per_epoch=len(train_loader))
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            train_loader = self.datamodule.get_sample_loader(train_nid)
            model = self.BaseModelCass(config, steps_per_epoch=len(train_loader))
            trainer.fit(model, train_dataloaders=train_loader)
        return model

    def train_k_fold_model(self):
        model_list = []
        for fold_idx, train_nid, val_nid in self.generate_kfold_nid():
            model = self.train_model(self.config, fold_idx, train_nid, val_nid)
            model_list.append(model)
        return model_list


def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    start_time = dt.datetime.now()
    graph, node_feat, labels = load_graph(config.data_dir, config.etypes)
    train_nid, val_nid, test_nid = load_default_split(config.data_dir)
    device = torch.device('cuda:{}'.format(config.gpu_id))

    datamodule = MaxpDataModule(config, (graph, node_feat, labels), (train_nid, val_nid, test_nid), device)
    kf_trainer = KFoldGNNTrainer(config, datamodule, MaxpLightning)
    model_list = kf_trainer.train_k_fold_model()
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))



# class KFoldGNN():
#     def __init__(self, config, datamodule, base_model_calss, device):
#         super().__init__()
#         self.config = config
#         self.datamodule = datamodule
#         self.default_train_nid, self.default_val_nid, self.default_predict_nid = datamodule.train_nid, datamodule.val_nid, datamodule.predict_nid
#         self.BaseModelCass = base_model_calss
#         self.device = device
        
#         self.cv = config.cv
#         self.random_state = config.random_state
#         self.ensembler = Ensembler(config.ensemble_type)
        

#     def train_model(self, config, fold_idx, train_nid, val_nid=None):
#         tr_args = get_trainer_args(config)
#         tr_args['logger'] = pl.loggers.TensorBoardLogger(
#                                 save_dir=f'./logs/ensemble_{config.training_mode}', name=config.name, version=f'fold_{fold_idx}',
#                                 default_hp_metric=False,
#                             )
#         trainer = pl.Trainer(
#             **tr_args
#         )
#         if val_nid is not None:
#             train_loader = self.datamodule.get_sample_loader(train_nid)
#             val_loader = self.datamodule.get_sample_loader(val_nid)
#             model = self.BaseModelCass(config, steps_per_epoch=len(train_loader))
#             trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#         else:
#             train_loader = self.datamodule.get_sample_loader(train_nid)
#             model = self.BaseModelCass(config, steps_per_epoch=len(train_loader))
#             trainer.fit(model, train_dataloaders=train_loader)
#         return model

#     def train_k_fold_model(self, nid, k):
#         nid = nid.cpu().numpy()
#         labels = self.datamodule.graph.ndata['labels'].cpu().numpy()[nid]
#         skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
#         model_list = []
#         for fold_idx, (nid_train_idx, nid_test_idx) in enumerate(skf.split(nid, labels)):
#             train_nid = torch.from_numpy(nid[nid_train_idx]).to(self.device)
#             val_nid = torch.from_numpy(nid[nid_test_idx]).to(self.device)
#             model = self.train_model(self.config, fold_idx, train_nid, val_nid)
#             model_list.append(model)
#         return model_list

#     def get_single_model_predictions(self, model, predict_nid):
#         loader = self.datamodule.get_inference_loader(predict_nid)
#         nids, preds, labels = graph_predict_with_labels(model, loader, self.device)
#         return nids, preds, labels

#     def model_ensemble(self):
#         if self.config.training_mode == 'inference':
#             train_nid = torch.cat([self.default_train_nid, self.default_val_nid], dim=-1)
#             predict_nid = self.default_predict_nid
#             all_nid = torch.cat([train_nid, predict_nid], dim=-1)

#             model_0 = self.train_model(self.config, 'default', self.default_train_nid, self.default_val_nid)
#             model_list = self.train_k_fold_model(train_nid, k=self.cv)
#             model_list.append(model_0)
#             res_list = []
#             for model in model_list:
#                 nids, preds, labels = self.get_single_model_predictions(model, all_nid)
#                 res_list.append((nids, preds, labels))
#             self.ensembler.init_data(res_list)
#             self.ensembler.fit(train_nid.cpu().numpy())
#             nids, preds = self.ensembler.predict(predict_nid.cpu().numpy())
#             save_submission(nids, preds, 
#                 filename='{}_{}'.format(self.config.name, self.config.version), 
#                 data_dir=self.config.data_dir, sub_dir=self.config.sub_dir
#             )

#         elif self.config.training_mode == 'evaluate':
#             train_val_nid = self.default_train_nid
#             test_nid = self.default_val_nid
#             all_nid = torch.cat([train_val_nid, test_nid], dim=-1)

#             model_list = self.train_k_fold_model(train_val_nid, k=self.cv)
#             res_list = []
#             for model in model_list:
#                 nids, preds, labels = self.get_single_model_predictions(model, all_nid)
#                 res_list.append((nids, preds, labels))
#             self.ensembler.init_data(res_list)
#             self.ensembler.fit(train_val_nid.cpu().numpy())
#             ensemble_result_log = self.ensembler.evaluate(test_nid.cpu().numpy())
#             print(json.dumps(ensemble_result_log, indent=4))


# def main(config):
#     print('----------------config----------------')
#     print(json.dumps(vars(config), indent=4))
#     print('--------------------------------------')

#     start_time = dt.datetime.now()
#     graph, node_feat, labels = load_graph(config.data_dir, config.etypes)
#     tr_nid, val_nid, test_nid = load_default_split(config.data_dir)
#     device = torch.device('cuda:{}'.format(config.gpu_id))
#     datamodule = MaxpDataModule(config, (graph, node_feat, labels), (tr_nid, val_nid, test_nid), device)

#     kf_model = KFoldGNN(config, datamodule, MaxpLightning, device)
#     kf_model.model_ensemble()
#     end_time = dt.datetime.now()

#     print('----------------Finished----------------')
#     print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
#     print('Using {} seconds'.format((end_time-start_time).seconds))



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config = EasyDict(vars(args))
    main(config)

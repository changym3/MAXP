import os
import glob
import json
import pickle
import argparse
import datetime as dt
from pytorch_lightning.core import datamodule
from easydict import EasyDict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics import Accuracy

from srcs.utils.utils import load_default_split, load_graph, save_submission, get_parser
from srcs.utils.training import ModelPredictor
from srcs.GNN.gnn_lightning import MaxpDataModule, MaxpLightning


class MyLR(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num_channels))
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.params)
        
    def forward(self, X):
        weights = self.params.softmax(dim=0)
        weights = weights.view(1, -1, 1)
        logits = (X * weights).sum(dim=1)
        return logits


class Ensembler():
    def __init__(self, predictors, datamodule, ensemble_type):
        self.predictors = predictors
        self.datamodule = datamodule
        self.ensemble_type = ensemble_type
        self.need_fit = False

        if self.ensemble_type == 'linear_voting':
            self.net = NeuralNetClassifier(
                module=MyLR,
                module__num_channels=len(predictors),
                max_epochs=2,
                optimizer=optim.Adam,
                optimizer__lr = .005,
                criterion=torch.nn.CrossEntropyLoss,
                iterator_train__shuffle=True,
                train_split=None,
                # device=device,
            )
            self.need_fit = True
        
    def fit(self, train_nid):
        if not self.need_fit:
            return
        nodes, preds, labels = self.run_predictors(train_nid)
        self.ensembler_fit(preds, labels)

    def predict(self, predict_nid):
        nodes, preds, labels = self.run_predictors(predict_nid)
        ensemble_preds = self.ensembler_predict(preds)
        return nodes, ensemble_preds[-1]

    def evaluate(self, test_nid):
        nodes, preds, labels = self.run_predictors(test_nid)
        ensemble_preds = self.ensembler_predict(preds)
        eval_res_log = self.get_evaluate_result(ensemble_preds, labels)
        return eval_res_log

    def run_predictors(self, nid):
        res_list = [prdtr.predict(nid) for prdtr in self.predictors]
        preds = []
        for base_nodes, base_preds, base_labels in res_list:
            new_idx = base_nodes.argsort()
            preds.append(base_preds[new_idx])
        preds = torch.stack(preds, dim=0)
        nodes = base_nodes[new_idx]
        labels = base_labels[new_idx]
        
        self.predictors_raw_res = res_list
        self.predictors_res = (nodes, preds, labels)
        # nodes: N
        # preds: B*N*C
        # labels: N
        return nodes, preds, labels
    
    def ensembler_fit(self, preds, labels):
        if self.ensemble_type == 'avg':
            pass
        elif self.ensemble_type == 'linear_voting':
            X = preds.permute(1, 0, 2)
            y = labels
            self.net.fit(X, y)

    def ensembler_predict(self, preds):
        if self.ensemble_type == 'avg':
            ensemble_preds = preds.softmax(dim=-1).mean(dim=0)
        elif self.ensemble_type == 'linear_voting':
            X = preds.permute(1, 0, 2)
            y_proba = self.net.predict_proba(X)
            ensemble_preds = torch.from_numpy(y_proba)
        ensemble_preds = torch.cat([preds, ensemble_preds.unsqueeze(0)], dim=0)
        return ensemble_preds
 
    def get_evaluate_result(self, preds, labels):
        ensemble_result_log = {
            'ensemble': self._evaluate_acc(preds[-1], labels)
        }
        for i, preds in enumerate(preds[:-1]):
            ensemble_result_log[f'fold_{i}'] = self._evaluate_acc(preds, labels)
        return ensemble_result_log

    def _evaluate_acc(self, preds, labels):
        return tm.functional.accuracy(preds, labels).item()




def load_models_from_dir(model_dir, ModelClass):
    pattern = os.path.join(model_dir, '**/*.ckpt')
    ckpt_list = glob.glob(pattern, recursive=True)
    model_list = [ModelClass.load_from_checkpoint(ckpt_path) for ckpt_path in ckpt_list]
    return model_list
    
        



def main(config):

    start_time = dt.datetime.now()
    model_list = load_models_from_dir(config.model_dir, MaxpLightning)

    model_config = EasyDict(model_list[0].hparams)
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')
    graph, node_feat, labels = load_graph(model_config.data_dir, model_config.etypes)
    train_nid, val_nid, test_nid = load_default_split(model_config.data_dir)
    device = torch.device('cuda:{}'.format(model_config.gpu_id))
    datamodule = MaxpDataModule(model_config, (graph, node_feat, labels), (train_nid, val_nid, test_nid), device)
    predictor_list = [ModelPredictor(model, datamodule, device) for model in model_list]

    ensembler = Ensembler(predictor_list, datamodule, config.ensemble_type)
    # nids, preds, labels, = ensembler.ensembler_predict(datamodule.val_nid)
    if config.training_mode == 'inference':
        ensembler.fit(torch.cat((train_nid, val_nid), dim=-1))
        nids, preds = ensembler.predict(test_nid)
        save_submission(nids, preds, 
                filename='{}_{}'.format(config.name, config.version), 
                data_dir=config.data_dir, sub_dir=config.sub_dir
        )
    elif config.training_mode == 'evaluate':
        ensembler.fit(train_nid)
        ensemble_result_log = ensembler.evaluate(val_nid)
        print(json.dumps(ensemble_result_log, indent=4))

    
    end_time = dt.datetime.now()

    print('----------------Finished----------------')
    print('Finished at {}', dt.datetime.now().strftime('%m-%d %H:%M'))
    print('Using {} seconds'.format((end_time-start_time).seconds))



if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--model_dir', type=str, default='/home/changym/competitions/MAXP/logs/ensemble_inference/Transformer/')
    args = parser.parse_args()
    config = EasyDict(vars(args))
    main(config)

    # python -m srcs.ensemble --gpu_id 0 --model_dir /home/changym/competitions/MAXP/logs/ensemble_inference/default --training_mode evaluate

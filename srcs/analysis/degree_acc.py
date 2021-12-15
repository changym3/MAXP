
import argparse
import datetime as dt
import json
import os

import pandas as pd
import sklearn.metrics as skm

import torch

from easydict import EasyDict

from srcs.utils.utils import get_trainer, get_parser
from srcs.utils.data import GraphDataset, NodeIDDataModule, Prediction
from srcs.MLP.mlp_model import MLPModel


def generate_degree_df(g, prediction, nids):
    df = pd.DataFrame(index=range(len(nids)))
    df = df.reset_index()
    df['labels'] = prediction.labels[nids].numpy()
    df['preds'] = prediction.predictions[nids].argmax(1).numpy()
    indg = g.in_degrees(nids).numpy()
    otdg = g.out_degrees(nids).numpy()
    indg[indg > 100] = 101
    otdg[otdg > 100] = 101
    df['in_degrees'] = indg
    df['out_degrees'] = otdg
    return df


def print_log_from_df(df):
    acc_res = df.groupby('in_degrees').apply(lambda x: skm.accuracy_score(x['labels'], x['preds']))
    count_res = df.groupby('in_degrees')['index'].count()
    count_sum = count_res.cumsum()

    print('There are total {} nodes'.format(count_sum.max()))
    nds = [3, 5, 10, 15, 20]
    for i in nds:
        print('There are total {} nodes ({:.2%}) with degree less than {}'.format(count_sum[i], count_sum[i]/count_sum.max(), i))
    print('--------------------------------------------')
    for i in range(0, 20):
        print("There are {} {}-degree nodes ({:.2%}) , their accuracy is {:.4f}".format(count_res[i], i, count_sum[i]/count_sum.max(), acc_res[i]))

    print('--------------------------------------------')
    nds = [30, 50, 70, 90]
    for i in nds:
        print("There are {} {}-degree nodes ({:.2%}), their accuracy is {:.4f}".format(count_res[i], i, count_sum[i]/count_sum.max(), acc_res[i])) 



def main(config):
    print('----------------config----------------')
    print(json.dumps(vars(config), indent=4))
    print('--------------------------------------')

    data_dir = '/home/changym/competitions/MAXP/official_data/processed_data/'
    etypes = ['bicited']
    dataset = GraphDataset(data_dir, etypes)

    pred_path = os.path.join(config.model_dir, 'prediction.pkl')
    prediction = Prediction.load_from_path(pred_path)

    stage2nid = {
        'train': dataset.train_nid.numpy(),
        'val': dataset.val_nid.numpy(),
    }
    for stage in config.stages:
        acc_res = generate_degree_df(dataset.graph, prediction, stage2nid[stage])
        print_log_from_df(acc_res)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./logs/sign_v1/base_L4/')
    parser.add_argument('--stages', nargs='+', default=['val'])
    args = parser.parse_args()
    config = EasyDict(vars(args))

    main(config)


# python -m srcs.analysis.degree_acc --stages val
# python -m srcs.analysis.degree_acc --stages val --model_dir /home/changym/competitions/MAXP/logs/MLP_v1/base
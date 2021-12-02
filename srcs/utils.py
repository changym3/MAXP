#-*- coding:utf-8 -*-

"""
    Utilities to handel graph data
"""

import argparse
import os
import pickle
import random
import tqdm
import dgl
import numpy as np
import pandas as pd
import torch


def load_graph(base_path, etypes):
    data_dict = {}
    df_path = os.path.join(base_path, 'edge_df.feather')
    cite_df = pd.read_feather(df_path)
    if 'cite' in etypes:
        data_dict[('paper', 'cite', 'paper')] = (cite_df.src_nid.values, cite_df.dst_nid.values)
    if 'cited' in etypes:
        data_dict[('paper', 'cited', 'paper')] = (cite_df.dst_nid.values, cite_df.src_nid.values)
    if 'bicited' in etypes:
        srcs = np.concatenate((cite_df.src_nid.values, cite_df.dst_nid.values))
        dsts = np.concatenate((cite_df.dst_nid.values, cite_df.src_nid.values))
        data_dict[('paper', 'bicited', 'paper')] = (srcs, dsts)
    graph = dgl.heterograph(data_dict)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)
    labels = torch.from_numpy(label_data['label']).long()
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = torch.from_numpy(features)
    graph.create_formats_()
    return graph, node_feat, labels


def save_submission(nodes, preds, filename, data_dir, sub_dir):
    label_array = np.array(list('ABCDEFGHIJKLMNOPQRSTUVW'))
    node_labels = label_array[preds.argmax(dim=1).numpy()]

    nodes_df = pd.read_csv(os.path.join(data_dir, 'IDandLabels.csv'), dtype={'Label':str})
    paper_array = nodes_df.paper_id.values
    node_ids = paper_array[nodes]

    test_submission = pd.DataFrame({'id': node_ids, 'label': node_labels})
    os.path.join(sub_dir, )
    sub_file_name = './submissions/{}.csv'.format(filename)
    test_submission.to_csv(sub_file_name, index=False)
    print('Finish saving submission to {}'.format(sub_file_name))


def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
import os
import pickle
import numpy as np
import pandas as pd
import torch

import dgl

import pytorch_lightning as pl





# class MyLoader(dgl.dataloading.NodeDataLoader):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def set_epoch(self, epoch):
#         if self.use_scalar_batcher:
#             self.scalar_batcher.set_epoch(epoch)
#         else:
#             self.dist_sampler.set_epoch(epoch)
class MaxpDataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(config['fanout'])
        self.graph = self.load_graph(config['DATA_PATH'], config['etypes']).to(device)
        train_nid, val_nid, test_nid = self.split_labels(self.config['DATA_PATH'])
        self.train_nid, self.val_nid, self.test_nid = train_nid.to(device), val_nid.to(device), test_nid.to(device)

    def train_dataloader(self):
        loader = dgl.dataloading.NodeDataLoader(self.graph,
                                              self.train_nid,
                                              self.sampler,
                                              device=self.device,
                                              batch_size=self.config['batch_size'],
                                              shuffle=True,
                                              drop_last=False,
                                              num_workers=self.config['num_workers']
                                              )
        loader.set_epoch = None
        return loader
    
    def val_dataloader(self):
        loader = dgl.dataloading.NodeDataLoader(self.graph,
                                              self.val_nid,
                                              self.sampler,
                                              device=self.device,
                                              batch_size=self.config['batch_size'],
                                              shuffle=True,
                                              drop_last=False,
                                              num_workers=self.config['num_workers']
                                              )
        loader.set_epoch = None
        return loader
    
    def describe(self):
        print('################ Graph info: ###############')
        print(self.graph)
        print('################ Label info: ################')
        print('Total labels (including not labeled): {}'.format(self.graph.ndata['labels'].shape[0]))
        print('               Training label number: {}'.format(self.train_nid.shape[0]))
        print('             Validation label number: {}'.format(self.val_nid.shape[0]))
        print('                   Test label number: {}'.format(self.test_nid.shape[0]))
        print('################ Feature info: ###############')
        print('Node\'s feature shape:{}'.format(self.graph.ndata['features'].shape))
    
    def load_graph(self, base_path, etypes=['cite', 'cited']):
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
            # cited_df = cite_df.rename(columns={'src_nid': 'dst_nid', 'dst_nid': 'src_nid'})
            # related_df = pd.concat([cite_df, cited_df], axis=0)
            # data_dict[('paper', 'cited', 'paper')] = (related_df.dst_nid.values, related_df.src_nid.values)
        graph = dgl.heterograph(data_dict)

        with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
        labels = torch.from_numpy(label_data['label']).long()
        features = np.load(os.path.join(base_path, 'features.npy'))
        node_feat = torch.from_numpy(features)

        graph.ndata['features'] = node_feat
        graph.ndata['labels'] = labels
        for et in etypes:
            graph = dgl.add_self_loop(graph, etype=et)
        graph.create_formats_()
        return graph

    def split_labels(self, base_path, splits='default'):
        with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
            label_data = pickle.load(f)
        labels = label_data['label']
        tr_label_idx = label_data['tr_label_idx']
        val_label_idx = label_data['val_label_idx']
        test_label_idx = label_data['test_label_idx']
        if splits == 'default':
            tr_label_idx = torch.from_numpy(tr_label_idx).long()
            val_label_idx = torch.from_numpy(val_label_idx).long()
            test_label_idx = torch.from_numpy(test_label_idx).long()
            pass
        elif isinstance(splits, list) and len(splits) == 2:
            all_label_idx = np.concatenate([tr_label_idx, val_label_idx])
            pass
        return tr_label_idx, val_label_idx, test_label_idx
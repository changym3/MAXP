import os
import pickle
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import pytorch_lightning as pl
from torchmetrics import Accuracy

from .gnn_model import HetGraphModel
from ..model_utils import get_optimizer


def load_model(ckpt_path):
    ckpt = MaxpLightning.load_from_checkpoint(ckpt_path)
    return ckpt


@torch.no_grad()
def graph_predict(model, datamodule, device):
    loader = datamodule.predict_dataloader()
    model = model.to(device)
    model.eval()
    nodes = []
    preds = []
    with tqdm.tqdm(loader) as tq:
        for batch in tq:
            _, output_nodes, _ = batch
            batch_pred = model(batch)
            nodes.append(output_nodes)
            preds.append(batch_pred)
        tq.set_description("Inference:")
    nodes = torch.cat(nodes, dim=0).cpu()
    preds = torch.cat(preds, dim=0).cpu()
    return nodes, preds



class MaxpLightning(pl.LightningModule):
    def __init__(self, config, steps_per_epoch=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.steps_per_epoch = steps_per_epoch
            
        self.gnn = self.configure_gnn()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
    def forward(self, batch):
        input_nodes, output_nodes, mfgs = batch
        # batch_inputs = mfgs[0].srcdata['features']
        # batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.gnn(mfgs)
        return batch_pred
    
    def training_step(self, batch, batch_idx):
        batch_pred = self(batch)
        _, _, mfgs = batch
        batch_labels = mfgs[-1].dstdata['labels']
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.train_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_pred = self(batch)
        _, _, mfgs = batch
        batch_labels = mfgs[-1].dstdata['labels']
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("hp/loss", loss)
        self.log("hp/acc", self.val_acc)
        return loss
    
    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/acc": 0, "hp/loss": 0})
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        return get_optimizer(self.config, self.parameters(), self.trainer.datamodule.train_dataloader())
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode='max', save_top_k=1)
        return [model_checkpoint]
    
    def configure_gnn(self):
        model = HetGraphModel(self.config)
        return model


class MaxpDataModule(pl.LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.graph = self.load_graph(config['data_dir'], config['etypes']).to(device)
        train_nid, val_nid, predict_nid = self.split_labels(self.config['data_dir'])
        self.train_nid, self.val_nid, self.predict_nid = train_nid.to(device), val_nid.to(device), predict_nid.to(device)
        self.configure_dataloader(config, self.graph, self.train_nid, self.val_nid, self.predict_nid)


    def configure_dataloader(self, config, graph, train_nid, val_nid, predict_nid):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(config['fanouts'])
        predict_sampler = dgl.dataloading.MultiLayerNeighborSampler([fo*10 for fo in config['fanouts']])
        self.train_loader = dgl.dataloading.NodeDataLoader(graph, train_nid, sampler,
                                                        device=self.device,
                                                        batch_size=self.config['batch_size'],
                                                        shuffle=True,
                                                        drop_last=False,
                                                        num_workers=self.config['num_workers']
                                                        )
        self.val_loader = dgl.dataloading.NodeDataLoader(graph, val_nid, sampler,
                                                        device=self.device,
                                                        batch_size=self.config['batch_size'],
                                                        shuffle=True,
                                                        drop_last=False,
                                                        num_workers=self.config['num_workers']
                                                        )
        self.predict_loader = dgl.dataloading.NodeDataLoader(graph, predict_nid, predict_sampler,
                                                            device=self.device,
                                                            batch_size=self.config['batch_size'],
                                                            shuffle=True,
                                                            drop_last=False,
                                                            num_workers=self.config['num_workers']
                                                            )
        self.train_loader.set_epoch = None
        self.val_loader.set_epoch = None
        self.predict_loader.set_epoch = None

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader

    def predict_dataloader(self):
        return self.predict_loader
    
    def describe(self):
        print('################ Graph info: ###############')
        print(self.graph)
        print('################ Label info: ################')
        print('Total labels (including not labeled): {}'.format(self.graph.ndata['labels'].shape[0]))
        print('               Training label number: {}'.format(self.train_nid.shape[0]))
        print('             Validation label number: {}'.format(self.val_nid.shape[0]))
        print('                   Test label number: {}'.format(self.predict_nid.shape[0]))
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
        # for et in etypes:
        #     graph = dgl.add_self_loop(graph, etype=et)
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

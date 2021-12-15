import os
import pickle

import dgl
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import tqdm

from ..utils.nn import ChannelCombine, get_optimizer
from . import gnn_modules as gm


class MaxpLightning(pl.LightningModule):
    def __init__(self, config, steps_per_epoch=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.steps_per_epoch = steps_per_epoch
        
        num_hiddens = config['num_hiddens']
        num_classes = config['num_classes']
        self.gnn = self.configure_gnn()
        self.classification = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_classes)
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
    def forward(self, batch):
        input_nodes, output_nodes, mfgs = batch
        batch_emb = self.gnn(mfgs)
        batch_pred = self.classification(batch_emb)
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
        return get_optimizer(self.config, self.parameters(), self.steps_per_epoch)
    
    def configure_callbacks(self):
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1)
        return [model_checkpoint]
    
    def configure_gnn(self):
        model = HetGraphModel(self.config)
        return model

    @torch.no_grad()
    def predict(self, loader, device):
        model = self.to(device)
        model.eval()
        nodes = []
        preds = []
        labels = []
        with tqdm.tqdm(loader) as tq:
            for batch in tq:
                _, output_nodes, mfgs = batch
                batch_pred = model(batch)
                batch_labels = mfgs[-1].dstdata['labels']
                nodes.append(output_nodes)
                preds.append(batch_pred)
                labels.append(batch_labels)
        nodes = torch.cat(nodes, dim=0).cpu()
        preds = torch.cat(preds, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        return nodes, preds, labels

    


class HetGraphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(config['dropout'])
        self.activation = nn.LeakyReLU()

        num_feats = config['num_features']
        num_hiddens=config['num_hiddens']
        self.feat_reduce = nn.Sequential(
            nn.Linear(num_feats, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
        )
        for l in range(0, config['num_layers']):
            if config['gnn_model'] == 'MLP':
                self.layers.append(nn.Linear(num_hiddens, num_hiddens))
            else:
                self.layers.append(HetGraphLayer(config))
            self.skips.append(nn.Linear(num_hiddens, num_hiddens))
            self.norms.append(nn.BatchNorm1d(num_hiddens))
        self.layer_combine = ChannelCombine(num_hiddens, config['layer_combine'], config['num_layers'] + 1)


    def blocks_forward(self, blocks, idxs):
        if len(blocks) == 0:
            h = blocks[-1].dstdata['features']
            h = self.feat_reduce(h)
            return h
        else:
            if self.config['gnn_model'] == 'MLP':
                h = blocks[-1].dstdata['features']
                h = self.feat_reduce(h)
                for i in range(len(self.layers)):
                    h_res = self.skips[i](h)
                    h = self.layers[i](h)
                    h += h_res
                    h = self.norms[i](h)
                    h = self.activation(h)
                    h = self.dropout(h)
                return h
            else:
                h = blocks[0].srcdata['features']
                h = self.feat_reduce(h)
                for i, block in zip(idxs, blocks):
                    h_res = self.skips[i](h[:block.num_dst_nodes()])
                    h = self.layers[i](block, h)
                    h += h_res
                    h = self.norms[i](h)
                    h = self.activation(h)
                    h = self.dropout(h)
                return h

    def forward(self, blocks):
        idxs = range(self.config['num_layers'])
        if self.config['layer_combine'] == 'last':
            h = self.blocks_forward(blocks, idxs)
        else:
            h_list = []
            for i in range(0, len(blocks)+1):
                h = self.blocks_forward(blocks[-i:], idxs[-i:])
                h_list.append(h)
            h = torch.stack(h_list, dim=0)
            h = self.layer_combine(h)
        return h


class HetGraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_hiddens = config['num_hiddens']
        etypes = config['etypes']
        gnn_model = config['gnn_model']
        num_heads = config['num_heads']
        feat_drop = config['feat_drop']
        attn_drop = config['attn_drop']

        if gnn_model == 'GCN':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.GraphConv(num_hiddens, num_hiddens)
                for rel in etypes}
            )
        elif gnn_model == 'GAT':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.GATConv(num_hiddens, num_hiddens // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
                for rel in etypes}
            )
        elif gnn_model == 'GATv2':
            self.mods = nn.ModuleDict(
                {rel: gm.GATv2Conv(num_hiddens, num_hiddens // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
                for rel in etypes}
            )
        elif gnn_model == 'SelectGAT':
            self.mods = nn.ModuleDict(
                {rel: gm.SelectGATConv(num_hiddens, num_hiddens // num_heads, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
                for rel in etypes}
            )
        elif gnn_model == 'Transformer':
            self.mods = nn.ModuleDict(
                {rel: gm.TransformerConv(num_hiddens, num_hiddens, num_heads, feat_drop=feat_drop, attn_drop=attn_drop)
                for rel in etypes}
            )
            
        elif gnn_model == 'SAGE':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.SAGEConv(num_hiddens, num_hiddens, aggregator_type='mean')
                for rel in etypes}
            )
        elif gnn_model == 'LGC':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.LightConv(num_hiddens, num_hiddens, norm='both')
                for rel in etypes})

        self.het_combine = ChannelCombine(num_hiddens, config['het_combine'], len(etypes))

        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)  

    def obtain_rel_embs(self, conv, g, x):
        if g.num_edges() == 0:
            return torch.zeros((g.num_dst_nodes(), self.out_feats), device=g.device)
        else:
            if 'GAT' in self.config['gnn_model']:
                return conv(g, x).flatten(1)
            else:
                return conv(g, x)
        
    def obtain_het_embs(self, g, x):
        outputs = []
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            dstdata = self.obtain_rel_embs(self.mods[etype], rel_graph, x)       
            outputs.append(dstdata)
        rel_embs = torch.stack(outputs, 0)
        return rel_embs
        
    def forward(self, g, x):
        rel_embs = self.obtain_het_embs(g, x)
        res = self.het_combine(rel_embs)
        return res



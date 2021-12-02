import torch
import torch.nn as nn
import dgl 
import torch.nn.functional as F
from ..model_utils import ChannelCombine
from .gnn_modules import GATv2Conv


class HetGraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_hiddens = config['num_hiddens']
        etypes = config['etypes']
        gnn_model = config['gnn_model']

        if gnn_model == 'GCN':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.GraphConv(num_hiddens, num_hiddens)
                for rel in etypes}
            )
        elif gnn_model == 'GAT':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.GATConv(num_hiddens, num_hiddens // config['num_heads'], config['num_heads'], feat_drop=config['feat_drop'], attn_drop=config['attn_drop'])
                for rel in etypes}
            )
        elif gnn_model == 'GATv2':
            self.mods = nn.ModuleDict(
                {rel: GATv2Conv(num_hiddens, num_hiddens // config['num_heads'], config['num_heads'], feat_drop=config['feat_drop'], attn_drop=config['attn_drop'])
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
        num_classes = config['num_classes']

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
        self.classification = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.BatchNorm1d(num_hiddens),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(num_hiddens, num_classes)
        )
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

        h = self.classification(h)
        return h




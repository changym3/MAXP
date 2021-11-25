import torch
import torch.nn as nn
import dgl 
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn


class HetGraphLayer(nn.Module):
    def __init__(self, config, in_feats, out_feats):
        super().__init__()
        self.config = config
        etypes = config.etypes
        gnn_model = config.gnn_model

        if gnn_model == 'HetGCN':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.GraphConv(in_feats, out_feats)
                for rel in etypes}
            )
        elif gnn_model == 'HetGAT':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.GATConv(in_feats, out_feats // config.num_heads, config.num_heads, residual=True)
                for rel in etypes}
            )
        elif gnn_model == 'HetSAGE':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.SAGEConv(in_feats, out_feats, aggregator_type='mean')
                for rel in etypes}
            )
        elif gnn_model == 'HetLGC':
            self.mods = nn.ModuleDict(
                {rel: dgl.nn.LightConv(in_feats, out_feats, norm='both')
                for rel in etypes})
            
        self.het_pooling = Pooling(out_feats, config.het_combine, len(etypes))
        
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        # for _, v in self.mods.items():
        #     set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
        #     if callable(set_allow_zero_in_degree_fn):
        #         set_allow_zero_in_degree_fn(True)  

    def obtain_rel_embs(self, conv, g, x):
        if self.config.gnn_model == 'HetGAT':
            return conv(g, x).flatten(1)
        else:
            return conv(g, x)

        
    def obtain_het_embs(self, g, x):
        outputs = []
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            dstdata = self.obtain_rel_embs(self.mods[etype], rel_graph, x)       
            outputs.append(dstdata)
        rel_embs = torch.stack(outputs)
        return rel_embs
        
    def forward(self, g, x):
        rel_embs = self.obtain_het_embs(g, x)
        res = self.het_pooling(rel_embs)
        return res
    
    # @torch.no_grad()
    # def inference_attention(self, g, x):
    #     rel_embs = self.obtain_rel_embs(g, x)
    #     return self.attn.inference_attention(rel_embs)




class Pooling(nn.Module):
    def __init__(self, dim_hiddens, pool_type, num_channels=None):
        super().__init__()
        self.dim_hiddens = dim_hiddens
        self.dim_query = dim_hiddens
        self.pool_type = pool_type
        self.num_channels = num_channels

        if pool_type == 'mean':
            pass
        elif pool_type == 'attn':
            self.attention = nn.Sequential(
                nn.Linear(dim_hiddens, dim_hiddens), nn.Tanh(), nn.Linear(dim_hiddens, 1, bias=False)
            )
        elif pool_type == 'transform':
            self.transformation = nn.Linear(dim_hiddens * num_channels, dim_hiddens)
    
    def forward(self, x):
        # x \in (C, N, d)
        if self.pool_type == 'attn':
            attn = F.softmax(self.attention(x), dim=0)
            res = (attn * x).sum(dim=0)
        elif self.pool_type == 'mean':
            res = x.mean(dim=0)
        elif self.attn_type == 'transform':
            # (C, N, d) to (N, C, d) to (N, C*d)
            x = x.permute(1, 0, 2).reshape(-1, self.num_channels * self.dim_hiddens)
            res = self.transformation(x)
        return res
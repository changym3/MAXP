import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def label_propagation(g, num_layers, labels, alpha, norm='both'):
    r"""
        alpha is the propagation coefficient.
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    """
    with g.local_scope():
        # labels = g.ndata['labels']
        labels = labels.clone().detach()
        labels[(labels == -1)] = (labels.max() + 1)
        labels = F.one_hot(labels).float()
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(labels.device).unsqueeze(dim=1)

        residual = (1 - alpha) * labels
        
        y = labels
        for _ in range(num_layers):
            # Assume the graphs to be undirected
            g.ndata['h'] = y * norm
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

            residual = (1 - alpha) * y
            aggr = alpha * g.ndata.pop('h')
            if norm == 'both':
                aggr = aggr * norm
            y =  residual + aggr
            y = torch.clamp(y, 0., 1.)
    return y


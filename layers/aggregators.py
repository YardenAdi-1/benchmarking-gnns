import torch
from torch import nn
import numpy as np

from layers.auxiliary_layers import MLP


class ConvAggregator(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = out_dim

        hidden_dim = hidden_dim_1 * hidden_dim_2
        self.affine_layer = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=True)

        self.mlp = MLP(in_size=hidden_dim, hidden_size=hidden_dim,
                       out_size=out_dim, layers=1,
                       mid_activation='relu', last_activation='none')

    def pretrans_edges(self, edges):
        pre = self.affine_layer(edges.src['h'])
        pre = pre.reshape(-1, self.hidden_dim_1, self.hidden_dim_2)
        return {'e': pre}

    def normalization(self, h):
        n_features = h.shape[-1] * h.shape[-2]
        n_neighbors = h.shape[1]
        normalization = np.sqrt(n_features ** (n_neighbors - 1))
        return h / normalization

    def conv_2d(self, h):
        h = torch.fft.fft2(h, dim=(-2, -1))
        h = torch.prod(h, dim=1, keepdim=True)
        h = torch.fft.ifft2(h, dim=(-2, -1))
        h = torch.real(h)
        return h

    def message_func(self, edges):
        return {'m': edges.data['e']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['m']
        h = self.conv_2d(h)
        return {'h': h}

    def posttrans(self, h):
        bs = h.shape[0]
        h = self.normalization(h)
        h = h.reshape(bs, -1)
        h = self.mlp(h)
        return h

    def forward(self, g, feature):
        g.ndata['h'] = feature

        # pre transformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)

        # post transformation
        h = self.posttrans(g.ndata['h'])

        return h


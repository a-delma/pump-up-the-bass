import torch
from gae import VariationalGCNEncoder as Enc
from node_classifier import simpleClassifer


class GCN(torch.nn.Module):
    def __init__(self, in_channel, inner_dim = 600):
        super().__init__()
        self.gae = Enc(in_channel, inner_dim)
        self.head = simpleClassifer(inner_dim)
    def forward(self, x, edge_index):
        # print(x.size())
        # print(edge_index)
        mu, _ = self.gae(x, edge_index)
        # print(mu.size())
        return self.head(mu)
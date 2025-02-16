
import torch.nn as nn

from torch_geometric.nn import MessagePassing


class MP_BP_APPR(MessagePassing):
    def __init__(self, constant):
        super().__init__(aggr = 'add')
        self.constant = constant
    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return x + self.constant * out
    def message(self, x_j):
        return x_j
    def update(self, aggr_out):
        return aggr_out

class AGG_BP_APPR(nn.Module):
    def __init__(self, weight):
        super(AGG_BP_APPR, self).__init__()
        self.weight = weight
        self.mp1 = MP_BP_APPR(self.weight)
    def forward(self, x, edge_index):
        x = self.mp1(x, edge_index)
        return x
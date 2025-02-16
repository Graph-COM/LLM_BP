

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

# the code is built upon Graph Belief Propagation Networks, Junteng Jia et al, 2021,
# the original github link is: https://github.com/000Justin000/GBPN

def log_normalize(log_x):
    return log_x - torch.logsumexp(log_x, -1, keepdim=True)

def degree(index, num_nodes, weight=None):
    out = torch.zeros(num_nodes, device=index.device)
    weight = weight if (weight is not None) else torch.ones(index.shape[0], device=index.device)
    return out.scatter_add_(0, index, weight)

def get_scaling(deg0, deg1):
    assert deg0.shape == deg1.shape
    scaling = torch.ones_like(deg0)
    scaling[deg1 != 0] = (deg0 / deg1)[deg1 != 0]
    return scaling


class BPConv(MessagePassing):

    def __init__(self, H):
        super(BPConv, self).__init__(aggr='add')
        self.H = torch.log(H)

    def forward(self, x, edge_index, edge_weight, info):
        # x has shape [N, n_channels]
        # edge_index has shape [2, E]
        # info has 4 fields: 'log_b0', 'log_msg_', 'edge_rv', 'msg_scaling'
        return self.propagate(edge_index, edge_weight=edge_weight, x=x, info=info)

    def message(self, x_j, edge_weight, info):
        # x_j has shape [E, n_channels]
        if info['log_msg_'] is not None:
            x_j = x_j - info['log_msg_'][info['edge_rv']]
        logC = self.H.unsqueeze(0) * edge_weight.unsqueeze(-1).unsqueeze(-1)
        log_msg_raw = torch.logsumexp(x_j.unsqueeze(-1) + logC, dim=-2)
        if info['msg_scaling'] is not None:
            log_msg_raw = log_msg_raw * info['msg_scaling'].unsqueeze(-1)
        log_msg = log_normalize(log_msg_raw)
        info['log_msg_'] = log_msg
        return log_msg

    def update(self, agg_log_msg, info):
        log_b_raw = info['log_b0'] + agg_log_msg
        log_b = log_normalize(log_b_raw)
        return log_b
    
    
class AGG_BP(nn.Module):
    def __init__(self, H):
        super(AGG_BP, self).__init__()
        self.bp_conv = BPConv(H)

    def forward(self, x, edge_index, edge_weight, edge_rv, deg):
        log_b0 = x
        #msg_scaling = get_scaling(deg[edge_index[1]], deg[edge_index[1]])
        msg_scaling = None
        info = {'log_b0': log_b0, 'log_msg_': None, 'edge_rv': edge_rv, 'msg_scaling': msg_scaling}
        log_b_ = log_b0
        for _ in range(5):
            log_b = self.bp_conv(log_b_, edge_index, edge_weight, info)
            log_b_ = log_b
            
        return log_b_
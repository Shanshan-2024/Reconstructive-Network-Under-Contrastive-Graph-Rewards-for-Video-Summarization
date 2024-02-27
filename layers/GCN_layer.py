import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN_Layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.act = nn.ReLU()

    def forward(self, x, adj):
        #x [34, 512]   adj [34, 34]   hidden [34, 256]  support [34, 256]   output(ri)  [34, 256]
        hidden = self.linear(x)
        support = torch.mm(adj, hidden)
        output = self.act(support)

        return output

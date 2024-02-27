import numpy as np
import torch
import torch.nn as nn
from layers.GCN_layer import GCN_Layer
from layers.Linear_layer import Linear, decoder
import torch.nn.functional as F

class GCN(nn.Module):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__()
        self.encode = GCN_Layer(1024, 256)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1)
        )
        self.fc3 = nn.Linear(256, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x, adj):
        # x  [34, 1024]
        # s_output 表示镜头的特征si r_output表示经过GCN卷积后的特征ri
        s_output = self.fc1(x)   #[34, 256]
        r_output = self.encode(x, adj)  #[34, 256]
        inputs = [s_output, r_output] #[34, 512]
        p = torch.sigmoid(self.fc2(torch.cat(inputs, dim=1)))
        # p = self.fc2(torch.cat(inputs, dim=1))
        h = self.fc3(r_output)
        output = self.fc4(h)
        if not self.training:
            '''
            需要改动draw_loss, plt_utils, torch.save
            '''
            # p = torch.sigmoid(p)
            output = torch.sigmoid(output)
        return p, r_output, h, output
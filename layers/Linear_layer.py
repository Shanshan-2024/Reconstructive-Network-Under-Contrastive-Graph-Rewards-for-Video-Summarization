import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    """
    Simple Linear layer with dropout.
    """
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear.forward(x)
        # hidden = F.dropout(hidden, training=self.training)
        # out = self.act(out)
        return out

class decoder(nn.Module):
    """
    简单的解码器模型
    """
    def __init__(self, in_features):
        super(decoder, self).__init__()
        self.linear1 = nn.Linear(in_features, 2)
        self.linear2 = nn.Linear(2, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        '''
        :param x: 最终的特征
        :return: h:B*2， out:B*1
        '''
        h = self.linear1.forward(x)
        out = self.linear2.forward(h)

        # hidden = F.dropout(hidden, training=self.training)
        # out = self.act(out)
        return out, h
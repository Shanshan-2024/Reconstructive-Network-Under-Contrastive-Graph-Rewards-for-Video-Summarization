import numpy as np
import torch
import torch.nn as nn
from layers.Linear_layer import Linear, decoder
import torch.nn.functional as F

class mlptomlp(nn.Module):
    """
    简单的线性模型
    """
    def __init__(self):
        super(mlptomlp, self).__init__()
        self.encode = Linear(1024, 128)
        self.decode = decoder(128)

    def forward(self, x):
        #x [34,1024]
        output = self.encode.forward(x) #[34,128]

        output, h = self.decode(output)  #output [34,1]  h [34,2]
        if not self.training:
            '''
            需要改动draw_loss, plt_utils, torch.save
            '''
            output = torch.sigmoid(output)
        return output, h
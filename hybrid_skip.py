import torch
import torch.nn as nn
import numpy as np
from kornia.filters import laplacian,gaussian_blur2d

#torch.cuda.empty_cache()

class HyperSkipConnection(nn.Module):
    def __init__(self, ch):
        super(HyperSkipConnection, self).__init__()

        ch_2 = int(ch/2)
        self.delta = torch.nn.Parameter(torch.randn(ch_2))
        self.delta.requires_grad = True
        self.epsilon = torch.nn.Parameter(torch.randn(ch_2))
        self.epsilon.requires_grad = True
        self.pointwise = nn.Conv2d(ch, ch_2, 1, 1, 0, bias=False)

    def multiply_(self, a,b):
        return (b.transpose(1,3)*a).transpose(1,3)
        
    def forward(self, e, d):
        e = self.pointwise(e)
        d = self.pointwise(d)
        e_l = gaussian_blur2d(e, (9,9), (1.5,1.5))
        d_h = laplacian(d, 9)

        h_d = self.multiply_(self.delta, d) + self.multiply_(1-self.delta, e_l)
        h_e = self.multiply_(self.epsilon, e) + self.multiply_(1-self.epsilon, d_h)
        F_hybrid = torch.cat([h_e, h_d],1)
        return F_hybrid


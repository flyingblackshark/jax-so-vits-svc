# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# from torch import nn
# from torch.nn import Conv1d
# from torch.nn.utils import weight_norm, remove_weight_norm
import jax.numpy as jnp
from flax import linen as nn
from vits import commons
from functools import partial
import flax
import jax
from typing import Tuple
# def init_weights(m, mean=0.0, std=0.01):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(mean, std)


# def get_padding(kernel_size, dilation=1):
#     return int((kernel_size*dilation - dilation)/2)
from jax.nn.initializers import normal as normal_init

class AMPBlock(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:Tuple[int]=(1, 3, 5)
    #train:bool = True
    def setup(self):
        super(AMPBlock, self).__init__()
        self.convs1 = [
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[0],
                               padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],
                               padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],
                               padding="SAME",kernel_init=normal_init(0.01))
        ]
        self.norms1=[nn.BatchNorm(scale_init=normal_init(0.01)) for i in range(3)]


        self.convs2 = [
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,
                               padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,
                               padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,
                               padding="SAME",kernel_init=normal_init(0.01))
        ]
        self.norms2=[nn.BatchNorm(use_running_average=not self.train, scale_init=normal_init(0.01)) for i in range(3)]


    def __call__(self, x,train = True):
        for c1, c2,n1,n2 in zip(self.convs1, self.convs2,self.norms1,self.norms2):
            xt = nn.leaky_relu(x, 0.1)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = n1(xt.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            xt = nn.leaky_relu(xt, 0.1)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = n2(xt.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            x = xt + x
        return x

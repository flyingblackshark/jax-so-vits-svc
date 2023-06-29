import jax.numpy as jnp
from flax import linen as nn
from vits import commons
from functools import partial
import flax
import jax
from typing import Tuple
from jax.nn.initializers import normal as normal_init
from .snake import snake
class AMPBlock(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple=(1, 3, 5)
    def setup(self):
       
        self.convs1 =[
            nn.Conv(self.channels,[ self.kernel_size], 1, kernel_dilation=self.dilation[0],kernel_init=normal_init(0.01)),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],kernel_init=normal_init(0.01)),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],kernel_init=normal_init(0.01))
        ]
        self.norms1 = [nn.BatchNorm(axis=-1,scale_init=normal_init(0.01),axis_name='num_devices') for i in range(3)]
        self.convs2 = [
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01)),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01)),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01))
        ]
        self.norms2 = [nn.BatchNorm(axis=-1,scale_init=normal_init(0.01),axis_name='num_devices') for i in range(3)]
        # total number of conv layers
        self.num_layers = len(self.convs1) + len(self.convs2)
    def __call__(self, x,train=True):
        for c1, c2,n1,n2 in zip(self.convs1, self.convs2,self.norms1,self.norms2):
            xt = n1(x.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            xt = snake(xt)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = n2(xt.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            xt = snake(xt)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x
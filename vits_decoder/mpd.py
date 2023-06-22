# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from vits import commons
class DiscriminatorP(nn.Module):
    hp:tuple
    period:int
    def setup(self):

        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope

        kernel_size = self.hp.mpd.kernel_size
        stride = self.hp.mpd.stride

        self.convs = [
            nn.Conv(features=64, kernel_size=(kernel_size, 1), strides=(stride, 1)),
            nn.Conv(features=128, kernel_size=(kernel_size, 1),strides= (stride, 1)),
            nn.Conv(features=256, kernel_size=(kernel_size, 1), strides=(stride, 1)),
            nn.Conv(features=512, kernel_size=(kernel_size, 1), strides=(stride, 1)),
            nn.Conv(features=1024, kernel_size=(kernel_size, 1), strides=1),
        ]
        self.norms=[nn.BatchNorm(scale_init=normal_init(0.1)) for i in range(5)]
        self.conv_post = nn.Conv(features=1, kernel_size=(3, 1), strides=1)
    

    def __call__(self, x,train=True):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, [(0,0),(0, 0),(0,n_pad)], "reflect")
            t = t + n_pad
        x = jnp.reshape(x,[b, c, t // self.period, self.period])
  
        for l,n in zip(self.convs,self.norms):
            x = l(x)
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            #x = commons.snake(x)
            x = n(x,use_running_average=not train)
          
            #x = nn.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])
        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.discriminators = [DiscriminatorP(self.hp, period) for period in self.hp.mpd.periods]
        

    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]

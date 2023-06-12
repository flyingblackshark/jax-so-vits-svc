# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
class DiscriminatorP(nn.Module):
    hp:tuple
    period:int
    def setup(self):
        #super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope
        #self.period = period

        kernel_size = self.hp.mpd.kernel_size
        stride = self.hp.mpd.stride
        #norm_f = weight_norm if self.hp.mpd.use_spectral_norm == False else spectral_norm

        self.convs = [
            nn.Conv(features=64, kernel_size=(kernel_size, 1), strides=(stride, 1), padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(features=128, kernel_size=(kernel_size, 1),strides= (stride, 1), padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(features=256, kernel_size=(kernel_size, 1), strides=(stride, 1), padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(features=512, kernel_size=(kernel_size, 1), strides=(stride, 1), padding="SAME",kernel_init=normal_init(0.01)),
            nn.Conv(features=1024, kernel_size=(kernel_size, 1), strides=1, padding="SAME",kernel_init=normal_init(0.01)),
        ]
        # self.norms = [
        #     nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.02)),
        #     nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.02)),
        #     nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.02)),
        #     nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.02)),
        #     nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.02))
        # ]
        self.norms=[nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01)) for i in range(5)]
        self.conv_post = nn.Conv(features=1, kernel_size=(3, 1), strides=1, padding="SAME",kernel_init=normal_init(0.02))
    

    def __call__(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, [(0,0),(0, 0),(0,n_pad)], "reflect")
            t = t + n_pad
        x = jnp.reshape(x,[b, c, t // self.period, self.period])
        x=x.transpose(0,1,3,2)
        for l,n in zip(self.convs,self.norms):
            x = l(x)
            x = n(x)
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x.transpose(0,1,3,2))
        x = self.conv_post(x)
        x=x.transpose(0,1,3,2)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])
        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        #super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = [DiscriminatorP(self.hp, period) for period in self.hp.mpd.periods]
        

    def __call__(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]

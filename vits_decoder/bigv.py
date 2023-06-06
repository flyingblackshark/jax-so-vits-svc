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

# def init_weights(m, mean=0.0, std=0.01):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(mean, std)


# def get_padding(kernel_size, dilation=1):
#     return int((kernel_size*dilation - dilation)/2)


class AMPBlock(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple[int]=(1, 3, 5)
    def setup(self):
        super(AMPBlock, self).__init__()
        self.convs1 = [
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[0],
                               padding="SAME"),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],
                               padding="SAME"),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],
                               padding="SAME")
        ]
        #self.convs1.apply(init_weights)

        self.convs2 = [
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,
                               padding="SAME"),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,
                               padding="SAME"),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,
                               padding="SAME")
        ]
        #self.convs2.apply(init_weights)

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = nn.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    # def remove_weight_norm(self):
    #     for l in self.convs1:
    #         remove_weight_norm(l)
    #     for l in self.convs2:
    #         remove_weight_norm(l)
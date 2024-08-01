import copy
import math
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from vits import commons
import jax


from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init

class WN(nn.Module):
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    gin_channels:int=0
    p_dropout:float=0

    def setup(self):
        assert self.kernel_size % 2 == 1
        in_layers = []
        res_skip_layers = []
        self.dropout_layer = nn.Dropout(rate=self.p_dropout)
        if self.gin_channels != 0:
            self.cond_layer = nn.Conv(features=2 * self.hidden_channels * self.n_layers,kernel_size=[1],kernel_init=nn.initializers.normal(),bias_init=nn.initializers.normal())
        for i in range(self.n_layers):
            dilation = self.dilation_rate**i
            in_layer = nn.Conv(
                features=2 * self.hidden_channels,
                kernel_size=[self.kernel_size],
                kernel_dilation=dilation,
                kernel_init=nn.initializers.normal(),
                bias_init=nn.initializers.normal()
            )
            in_layers.append(in_layer)

            # last one is not necessary
            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.hidden_channels
            else:
                res_skip_channels = self.hidden_channels

            res_skip_layer = nn.Conv(features=res_skip_channels, kernel_size=[1],bias_init=nn.initializers.normal(),kernel_init=nn.initializers.normal())
            res_skip_layers.append(res_skip_layer)
        self.res_skip_layers = res_skip_layers
        self.in_layers = in_layers
        
       
    def __call__(self, x, x_mask, g=None,train=True, **kwargs):
        output = jnp.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g.transpose(0,2,1)).transpose(0,2,1)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x.transpose(0,2,1)).transpose(0,2,1)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels,:]
            else:
                g_l = jnp.zeros_like(x_in)

            in_act = x_in + g_l
            t_act = nn.tanh(in_act[:, :self.hidden_channels, :])
            s_act = nn.sigmoid(in_act[:, self.hidden_channels:, :])
            acts = t_act * s_act
            acts = self.dropout_layer(acts,deterministic=not train)

            res_skip_acts = self.res_skip_layers[i](acts.transpose(0,2,1)).transpose(0,2,1)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts)
                x = jnp.where(x_mask,x,0)
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        output = jnp.where(x_mask,output,0)
        return output



class Flip(nn.Module):
    def __call__(self, x, *args, reverse=False, **kwargs):
        x = jnp.flip(x, 1)
        if not reverse:
            logdet = jnp.zeros(x.shape[0])
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(nn.Module):
    channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    p_dropout:float=0
    gin_channels:int=0
    mean_only:bool=False
    def setup(
        self
    ):
        assert self.channels % 2 == 0, "channels should be divisible by 2"
        self.half_channels = self.channels // 2

        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[1],dtype=jnp.float32,bias_init=nn.initializers.normal())
        # no use gin_channels
        self.enc = WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            gin_channels = self.gin_channels,
            p_dropout=self.p_dropout
        )
        self.post = nn.Conv(features= self.half_channels * (2 - self.mean_only), kernel_size=[1],kernel_init=constant_init(0.),bias_init=constant_init(0.),dtype=jnp.float32)


    def __call__(self, x, x_mask, g=None, reverse=False,train=True):

        x0, x1 = jnp.split(x, 2 , axis=1)
        h = self.pre(x0.transpose(0,2,1)).transpose(0,2,1) * x_mask
        h = self.enc(h, x_mask,g=g,train=train)
        stats = jnp.where(x_mask,self.post(h.transpose(0,2,1)).transpose(0,2,1),0)
        if not self.mean_only:
            m, logs = jnp.split(stats, 2, 1)
        else:
            m = stats
            logs = jnp.zeros_like(m)

        if not reverse:
            x1 = m + x1 * jnp.exp(logs) * x_mask
            x = jnp.concatenate([x0, x1], 1)
            logdet = jnp.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * jnp.exp(-logs) * x_mask
            x = jnp.concatenate([x0, x1], 1)
            return x
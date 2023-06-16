import copy
import math
import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
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
    p_dropout:float=0.

    def setup(self):
        assert self.kernel_size % 2 == 1
        in_layers = []#torch.nn.ModuleList()
        res_skip_layers = []#torch.nn.ModuleList()
        #self.dropout_layer = nn.Dropout(rate=self.p_dropout)

        if self.gin_channels != 0:
            self.cond_layer = nn.Conv(
                features=2 * self.hidden_channels * self.n_layers,kernel_size=[1])
            self.cond_layer_norm = nn.BatchNorm(scale_init=normal_init(0.01))
        in_layer_norms = []
        res_skip_layer_norms = []
        for i in range(self.n_layers):
            dilation = self.dilation_rate**i
            in_layer = nn.Conv(
                features=2 * self.hidden_channels,
                kernel_size=[self.kernel_size],
                kernel_dilation=dilation,
            )
            in_layers.append(in_layer)
            in_layer_norms.append(nn.BatchNorm(scale_init=normal_init(0.01)))
            # last one is not necessary
            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.hidden_channels
            else:
                res_skip_channels = self.hidden_channels

            res_skip_layer = nn.Conv(features=res_skip_channels, kernel_size=[1])
            res_skip_layers.append(res_skip_layer)
            res_skip_layer_norms.append(nn.BatchNorm(scale_init=normal_init(0.01)))
        self.res_skip_layers = res_skip_layers
        self.in_layers = in_layers
        self.in_layer_norms = in_layer_norms
        self.res_skip_layer_norms = res_skip_layer_norms

    def __call__(self, x, x_mask, g=None,train=True, **kwargs):
        #x = x.transpose(0,2,1)
        output = jnp.zeros_like(x)
        n_channels_tensor = [self.hidden_channels]

        if g is not None:
            g = self.cond_layer(g.transpose(0,2,1)).transpose(0,2,1)
            g = self.cond_layer_norm(g.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x.transpose(0,2,1)).transpose(0,2,1)
            x_in = self.in_layer_norms[i](x_in.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels,:]
            else:
                g_l = jnp.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            #acts = self.dropout_layer(acts,deterministic=not train)

            res_skip_acts = self.res_skip_layers[i](acts.transpose(0,2,1)).transpose(0,2,1)
            res_skip_acts = self.res_skip_layer_norms[i](res_skip_acts.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask



class Flip(nn.Module):
    def __call__(self, x, *args, reverse=False, **kwargs):
        x = jnp.flip(x, [1])
        logdet = jnp.zeros(x.shape[0])
        return x, logdet


class ResidualCouplingLayer(nn.Module):
    channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    p_dropout:float=0,
    gin_channels:int=0,
    mean_only:bool=False,
    #train:bool=True
    def setup(
        self
    ):
        assert self.channels % 2 == 0, "channels should be divisible by 2"
        self.half_channels = self.channels // 2

        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[1])
        # no use gin_channels
        self.enc = WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            p_dropout=self.p_dropout
        )
        self.post = nn.Conv(
            features= self.half_channels * (2 - self.mean_only), kernel_size=[1],kernel_init=constant_init(0.),bias_init=constant_init(0.))
        # SNAC Speaker-normalized Affine Coupling Layer
        self.snac = nn.Conv(features=2 * self.half_channels, kernel_size=[1])

    def __call__(self, x, x_mask, g=None, reverse=False,train=True):
        speaker = jnp.expand_dims(self.snac(g),-1)
        speaker_m, speaker_v = jnp.split(speaker,2, axis=1)  # (B, half_channels, 1)
        x0, x1 = jnp.split(x,  [self.half_channels] , axis=1)
        # x0 norm
        x0_norm = (x0 - speaker_m) * jnp.exp(-speaker_v) * x_mask
        h = self.pre(x0_norm.transpose(0,2,1)).transpose(0,2,1) * x_mask
        # don't use global condition
        h = self.enc(h, x_mask,train=train)
        stats = self.post(h.transpose(0,2,1)).transpose(0,2,1)* x_mask
        if not self.mean_only:
            m, logs = jnp.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = jnp.zeros_like(m)

        if not reverse:
            # x1 norm before affine xform
            x1_norm = (x1 - speaker_m) * jnp.exp(-speaker_v) * x_mask
            x1 = (m + x1_norm * jnp.exp(logs)) * x_mask
            x = jnp.concatenate([x0, x1], 1)
            # speaker var to logdet

            logdet = jnp.sum(logs * x_mask, [1, 2]) - jnp.sum(
                jnp.broadcast_to(speaker_v,(speaker_v.shape[0], speaker_v.shape[1], logs.shape[-1])) * x_mask, [1, 2])
      
            return x, logdet
        else:
            x1 = (x1 - m) * jnp.exp(-logs) * x_mask
            # x1 denorm before output
            x1 = (speaker_m + x1 * jnp.exp(speaker_v)) * x_mask
            x = jnp.concatenate([x0, x1], 1)
            # speaker var to logdet

            logdet = jnp.sum(logs * x_mask, [1, 2]) + jnp.sum(
                 jnp.broadcast_to(speaker_v,(speaker_v.shape[0], speaker_v.shape[1], logs.shape[-1])) * x_mask, [1, 2])
           
            return x, logdet
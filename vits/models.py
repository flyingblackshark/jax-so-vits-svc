
#import torch

#from torch import nn
#from torch.nn import functional as F
# from absl import app
# from absl import flags
import flax
from flax import linen as nn
from jax.nn.initializers import normal as normal_init
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
import tensorflow as tf
#import tensorflow_datasets as tfds

from vits import attentions
from vits import commons
from vits import modules
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator
#from vits.modules_grl import SpeakerClassifier

class TextEncoder(nn.Module):
    in_channels:int
    out_channels:int
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int
    p_dropout:float
    def setup(self):
        #super().__init__()
        #self.out_channels = out_channels
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[5], padding=2)
        self.pit = nn.Embed(256, self.hidden_channels,dtype=jnp.float32)
        self.enc = attentions.Encoder(
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout)
        self.proj = nn.Conv(features=self.out_channels * 2, kernel_size=[1])

    def __call__(self, x, x_lengths, f0):
        rng = random.PRNGKey(1234)
        x = x.transpose(0,2,1)  # [b, h, t]
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x.transpose(0,2,1)).transpose(0,2,1) * x_mask
        x = x + self.pit(f0).transpose(0, 2,1)
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x.transpose(0,2,1)).transpose(0,2,1) * x_mask
        m, logs = jnp.split(stats,2, axis=1) #self.out_channels, axis=1)
        z = (m + jax.random.normal(rng,m.shape) * jnp.exp(logs)) * x_mask
        return z, m, logs, x_mask, x


class ResidualCouplingBlock(nn.Module):
    channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    n_flows:int=4
    gin_channels:int=0
    def setup(
        self
    ):
        #super().__init__()
        flows = []#nn.ModuleList()
        for i in range(self.n_flows):
            flows.append(
                modules.ResidualCouplingLayer(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.n_layers,
                    gin_channels=self.gin_channels,
                    mean_only=True,
                )
            )
            flows.append(modules.Flip())
        self.flows=flows

    def __call__(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    # def remove_weight_norm(self):
    #     for i in range(self.n_flows):
    #         self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    in_channels:int
    out_channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    gin_channels:int=0,
    def setup(
        self
    ):
        #super().__init__()
        #self.out_channels = out_channels
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[1])
        self.enc = modules.WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv(features=self.out_channels * 2,kernel_size=[1])

    def __call__(self, x, x_lengths,g=None):
        rng = random.PRNGKey(1234)
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x.transpose(0,2,1)).transpose(0,2,1) * x_mask
        x = self.enc(x.transpose(0,2,1), x_mask, g=g).transpose(0,2,1)
        stats = self.proj(x).transpose(0,2,1) * x_mask
        m, logs = jnp.split(stats, 2, axis=1)
        z = (m + jax.random.normal(rng,m.shape) * jnp.exp(logs)) * x_mask
        return z, m, logs, x_mask

    # def remove_weight_norm(self):
    #     self.enc.remove_weight_norm()


class SynthesizerTrn(nn.Module):
    spec_channels : int
    segment_size : int
    hp:tuple
    def setup(
        self,
        #spec_channels,
       # segment_size,
       # hp
    ):
        #super().__init__()
        #self.segment_size = segment_size
        self.emb_g = nn.Dense(self.hp.vits.spk_dim, self.hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            self.hp.vits.ppg_dim,
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            self.hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        # self.speaker_classifier = SpeakerClassifier(
        #     self.hp.vits.hidden_channels,
        #     self.hp.vits.spk_dim,
        # )
        self.enc_q = PosteriorEncoder(
            self.spec_channels,
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=self.hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=self.hp.vits.spk_dim
        )
        self.dec = Generator(hp=self.hp)

    def __call__(self, ppg, pit, spec, spk, ppg_l, spec_l):
        rng = random.PRNGKey(1234)
        ppg = ppg + jax.random.normal(rng,ppg.shape)#torch.randn_like(ppg)  # Perturbation
        
        g = jnp.expand_dims(self.emb_g(spk),-1)
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)

        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size)
        audio = self.dec(spk, z_slice, pit_slice)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, reverse=True)
        # speaker
        #spk_preds = self.speaker_classifier(x)
        return audio, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r)#, spk_preds

    def infer(self, ppg, pit, spk, ppg_l):
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit)
        return o


class SynthesizerInfer(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference(self, ppg, pit, spk, ppg_l, source):
        rng = random.PRNGKey(1234)
        ppg = ppg + jax.random.normal(rng,ppg.shape) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
        z_p = m_p + jax.random.normal(rng,m_p.shape) * jnp.exp(logs_p) * 0.7  
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o

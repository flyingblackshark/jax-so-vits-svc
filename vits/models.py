

from flax import linen as nn
import jax.numpy as jnp
import jax
from vits import attentions
from vits import commons
from vits import modules
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator


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
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[5],dtype=jnp.float32,bias_init=nn.initializers.normal(),kernel_init=nn.initializers.normal())
        self.pit = nn.Embed(256, self.hidden_channels,dtype=jnp.float32,embedding_init=nn.initializers.normal(1.0))
        self.enc = attentions.Encoder(
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout)
        self.proj = nn.Conv(features=self.out_channels * 2, kernel_size=[1],dtype=jnp.float32,bias_init=nn.initializers.normal(),kernel_init=nn.initializers.normal())
    def __call__(self, x, x_lengths, f0,train=True):
        x = x.transpose(0,2,1)
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x.transpose(0,2,1)).transpose(0,2,1) * x_mask
        x = x + self.pit(f0).transpose(0,2,1)
        x = self.enc(jnp.where(x_mask,x,0), x_mask,train=train)
        stats = self.proj(x.transpose(0,2,1)).transpose(0,2,1) * x_mask
        m, logs = jnp.split(stats,2, axis=1)
        z = (m + jax.random.normal(self.make_rng('rnorms'),m.shape) * jnp.exp(logs)) * x_mask
        return z, m, logs, x_mask


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
        flows = []
        for i in range(self.n_flows):
            flows.append(
                modules.ResidualCouplingLayer(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.n_layers,
                    gin_channels=self.gin_channels,
                    mean_only=True
                )
            )
            flows.append(modules.Flip())
        self.flows=flows

    def __call__(self, x, x_mask, g=None, reverse=False,train=True):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse,train=train)
            return x
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse,train=train)
            return x


class PosteriorEncoder(nn.Module):
    in_channels:int
    out_channels:int
    hidden_channels:int
    kernel_size:int
    dilation_rate:int
    n_layers:int
    gin_channels:int=0
    def setup(
        self
    ):
        self.pre = nn.Conv(features=self.hidden_channels, kernel_size=[1],bias_init=nn.initializers.normal(),kernel_init=nn.initializers.normal())
        self.enc = modules.WN(
            self.hidden_channels,
            self.kernel_size,
            self.dilation_rate,
            self.n_layers,
            gin_channels=self.gin_channels
        )
        self.proj = nn.Conv(features=self.out_channels * 2,kernel_size=[1],bias_init=nn.initializers.normal(),kernel_init=nn.initializers.normal())

    def __call__(self, x, x_lengths,g=None,train=True):
        x = x.tranpose(0,2,1)
        rng = self.make_rng('rnorms')
        normal_key,rng = jax.random.split(rng,2)
        x_mask = jnp.expand_dims(commons.sequence_mask(x_lengths, x.shape[2]), 1)
        x = self.pre(x.transpose(0,2,1)).transpose(0,2,1)
        x = jnp.where(x_mask,x,0)
        x = self.enc(x, x_mask, g=g,train=train)
        stats = self.proj(x.transpose(0,2,1)).transpose(0,2,1)
        stats = jnp.where(x_mask,stats,0)
        m, logs = jnp.split(stats,2, axis=1)
        z = (m + jax.random.normal(normal_key,m.shape) * jnp.exp(logs))
        z = jnp.where(x_mask,z,0)
        return z, m, logs, x_mask
    
class SynthesizerTrn(nn.Module):
    spec_channels : int
    segment_size : int
    hp:tuple
    def setup(self):
        self.emb_g = nn.Embed(self.hp.data.n_speakers +1,self.hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            self.hp.vits.ppg_dim,
            self.hp.vits.inter_channels,
            self.hp.vits.hidden_channels,
            self.hp.vits.filter_channels,
            2,
            6,
            3,
            0.1
        )
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
            gin_channels=self.hp.vits.spk_dim,
        )
        self.dec = Generator(hp=self.hp)

    def __call__(self, ppg, pit, spec,spk, ppg_l, spec_l,train=True):
        g = self.emb_g(jnp.expand_dims(spk,-1)).transpose(0,2,1)
        z_ptemp, m_p, logs_p, _ = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit),train=train)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g,train=train)
        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z, pit, spec_l, self.segment_size,rng=self.make_rng('rnorms'))
        audio = self.dec(z_slice, pit_slice,train=train)
        z_p = self.flow(z, spec_mask, g=g,reverse=False,train=train)
        return audio, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, ppg, pit, spk, ppg_l):
        z_p, m_p, logs_p, ppg_mask = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit),train=False)
        g = self.emb_g(jnp.expand_dims(spk,-1)).transpose(0,2,1)
        z = self.flow(z_p, ppg_mask, g=g, reverse=True,train=False)
        o = self.dec(z * ppg_mask, f0=pit,train=False)
        return o

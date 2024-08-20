import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from vits.modules import ResBlock1
from .nsf import SourceModuleHnNSF
from jax.nn.initializers import normal as normal_init




class Generator(nn.Module):
    hp:tuple
    def setup(self):
        self.num_kernels = len(self.hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(self.hp.gen.upsample_rates)
        self.conv_pre = nn.WeightNorm(nn.Conv(features=self.hp.gen.upsample_initial_channel, kernel_size=[7], strides=[1]))
        self.scale_factor = np.prod(self.hp.gen.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.hp.data.sampling_rate)
        noise_convs = []
        ups = []
        for i, (u, k) in enumerate(zip(self.hp.gen.upsample_rates, self.hp.gen.upsample_kernel_sizes)):
            ups.append(
                    nn.WeightNorm(nn.ConvTranspose(
                        self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,)))
                )
            if i + 1 < len(self.hp.gen.upsample_rates):
                stride_f0 = np.prod(self.hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                noise_convs.append(
                    nn.Conv(
                        features=self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=[stride_f0]
                    )
                )
            else:
                noise_convs.append(
                    nn.Conv(features=self.hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1])
                )

        resblocks = []
        for i in range(len(ups)):
            ch = self.hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.hp.gen.resblock_kernel_sizes, self.hp.gen.resblock_dilation_sizes):
                resblocks.append(ResBlock1(ch, k, d))

        self.conv_post =  nn.WeightNorm(nn.Conv(features=1, kernel_size=[7], strides=1 , use_bias=False))
        self.ups = ups
        self.noise_convs = noise_convs
        self.resblocks = resblocks

    def __call__(self, x, f0,train=True):
        x = x + jax.random.normal(self.make_rng('rnorms'),x.shape)

        f0 = f0[:, None]
        B, H, W = f0.shape
        f0 = jax.image.resize(f0, shape=(B, H, W * self.scale_factor), method='nearest').transpose(0,2,1)
        har_source = self.m_source(f0,self.make_rng('rnorms'))
        har_source = har_source.transpose(0,2,1)
        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, 0.1)
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)
            x_source = self.noise_convs[i](har_source.transpose(0,2,1)).transpose(0,2,1)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        x = nn.leaky_relu(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.tanh(x) 
        return x
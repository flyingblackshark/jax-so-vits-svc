import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from .nsf import SourceModuleHnNSF

from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from vits import commons
# class SpeakerAdapter(nn.Module):
#     speaker_dim : int
#     adapter_dim : int
#     epsilon : int = 1e-5
#     def setup(self):
#         self.W_scale = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(1.),dtype=jnp.float32)
#         self.W_bias = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(0.),dtype=jnp.float32)


#     def __call__(self, x, speaker_embedding):
#         x = x.transpose(0,2,1)
#         mean = x.mean(axis=-1, keepdims=True)
#         var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
#         std = jnp.sqrt(var + self.epsilon)
#         y = (x - mean) / std
#         scale = self.W_scale(speaker_embedding)
#         bias = self.W_bias(speaker_embedding)
#         y *= jnp.expand_dims(scale,1)
#         y += jnp.expand_dims(bias,1)
#         y = y.transpose(0,2,1)
#         return y

class ResBlock1(nn.Module):
    channels:int
    kernel_size:int=3
    dilation:tuple=(1, 3, 5)
    def setup(self):
       
        self.convs1 =[
            nn.Conv(self.channels,[ self.kernel_size], 1, kernel_dilation=self.dilation[0],kernel_init=normal_init(0.01),bias_init=nn.initializers.normal()),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[1],kernel_init=normal_init(0.01),bias_init=nn.initializers.normal()),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=self.dilation[2],kernel_init=normal_init(0.01),bias_init=nn.initializers.normal())]
        self.convs2 = [
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01),bias_init=nn.initializers.normal()),
            nn.Conv( self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01),bias_init=nn.initializers.normal()),
            nn.Conv(self.channels, [self.kernel_size], 1, kernel_dilation=1,kernel_init=normal_init(0.01),bias_init=nn.initializers.normal())
        ]
        self.num_layers = len(self.convs1) + len(self.convs2)
        
    def __call__(self, x,train=True):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x,0.1)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = nn.leaky_relu(xt,0.1)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x

class Generator(nn.Module):
    hp:tuple
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def setup(self):
        self.num_kernels = len(self.hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(self.hp.gen.upsample_rates)
      
        # pre conv
        self.conv_pre = nn.Conv(features=self.hp.gen.upsample_initial_channel, kernel_size=[7], strides=[1],dtype=jnp.float32,bias_init=nn.initializers.normal())
        # nsf
        # self.f0_upsamp = nn.Upsample(
        #     scale_factor=np.prod(hp.gen.upsample_rates))
        self.scale_factor = np.prod(self.hp.gen.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.hp.data.sampling_rate)
        noise_convs = []
        # transposed conv-based upsamplers. does not apply anti-aliasing
        ups = []
        for i, (u, k) in enumerate(zip(self.hp.gen.upsample_rates, self.hp.gen.upsample_kernel_sizes)):
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            ups.append(
                    nn.ConvTranspose(
                        self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        (k,),
                        (u,),
                        kernel_init=normal_init(0.01),bias_init=nn.initializers.normal())
                )
            # nsf
            if i + 1 < len(self.hp.gen.upsample_rates):
                stride_f0 = np.prod(self.hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                noise_convs.append(
                    nn.Conv(
                        features=self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=[stride_f0],
                        dtype=jnp.float32,bias_init=nn.initializers.normal()
                    )
                )
            else:
                noise_convs.append(
                    nn.Conv(features=self.hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1],dtype=jnp.float32,bias_init=nn.initializers.normal())
                )

        resblocks = []
        for i in range(len(ups)):
            ch = self.hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.hp.gen.resblock_kernel_sizes, self.hp.gen.resblock_dilation_sizes):
                resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = nn.Conv(features=1, kernel_size=[7], strides=1 , use_bias=False,dtype=jnp.float32)
        self.ups = ups
        self.noise_convs = noise_convs
        self.resblocks = resblocks

    def __call__(self, x, f0,train=True):
        x = x + jax.random.normal(self.make_rng('rnorms'),x.shape)
        # nsf
        f0 = f0[:, None]
        B, H, W = f0.shape
        f0 = jax.image.resize(f0, shape=(B, H, W * self.scale_factor), method='nearest').transpose(0,2,1)
        har_source = self.m_source(f0,self.make_rng('rnorms'))
        har_source = har_source.transpose(0,2,1)
        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)
            # nsf
            x_source = self.noise_convs[i](har_source.transpose(0,2,1)).transpose(0,2,1)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        # post conv
        
        x = nn.leaky_relu(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        x = nn.tanh(x) 
        return x

    def inference(self, spk, x, har_source):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = nn.functional.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        x = nn.tanh(x)
        return x

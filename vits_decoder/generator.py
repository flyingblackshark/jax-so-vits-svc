# import torch
# import torch.nn as nn
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
# from torch.nn import Conv1d
# from torch.nn import ConvTranspose1d
# from torch.nn.utils import weight_norm
# from torch.nn.utils import remove_weight_norm

from .nsf import SourceModuleHnNSF
from .bigv import AMPBlock
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init

class SpeakerAdapter(nn.Module):
    speaker_dim : int
    adapter_dim : int
    epsilon : int = 1e-5
    def setup(self):
        #super(SpeakerAdapter, self).__init__()
        # self.speaker_dim = self.speaker_dim
        # self.adapter_dim = self.adapter_dim
        # self.epsilon = self.epsilon
        self.W_scale = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(1.))
        self.W_bias = nn.Dense(features=self.adapter_dim,kernel_init=constant_init(0.),bias_init=constant_init(0.))
        #self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.constant_(self.W_scale.weight, 0.0)
    #     torch.nn.init.constant_(self.W_scale.bias, 1.0)
    #     torch.nn.init.constant_(self.W_bias.weight, 0.0)
    #     torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def __call__(self, x, speaker_embedding):
        x = x.transpose(0,2,1)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.epsilon)
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= jnp.expand_dims(scale,1)
        y += jnp.expand_dims(bias,1)
        y = y.transpose(0,2,1)
        return y


class Generator(nn.Module):
    hp:tuple
    train:bool=True
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def setup(self):
        #super(Generator, self).__init__()
        #self.hp = hp
        self.num_kernels = len(self.hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(self.hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = SpeakerAdapter(self.hp.vits.spk_dim, self.hp.gen.upsample_input)
        # pre conv
        self.conv_pre = nn.Conv(features=self.hp.gen.upsample_initial_channel, kernel_size=[7], strides=[1], padding="SAME",kernel_init=normal_init(0.01))
        # nsf
        # self.f0_upsamp = nn.Upsample(
        #     scale_factor=np.prod(hp.gen.upsample_rates))
        self.scale_factor = np.prod(self.hp.gen.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=self.hp.data.sampling_rate)
        noise_convs = []#nn.ModuleList()
        #noise_conv_norms = []
        # transposed conv-based upsamplers. does not apply anti-aliasing
        ups = []#nn.ModuleList()
        #ups_norm = []
        for i, (u, k) in enumerate(zip(self.hp.gen.upsample_rates, self.hp.gen.upsample_kernel_sizes)):
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            ups.append(
                    nn.ConvTranspose(
                       features= self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                       kernel_size= k,
                        strides=[u],
                        padding="SAME",kernel_init=normal_init(0.01))
                )
            #ups_norm.append(nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01)))
            
            # nsf
            if i + 1 < len(self.hp.gen.upsample_rates):
                stride_f0 = np.prod(self.hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                noise_convs.append(
                    nn.Conv(
                        features=self.hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=[stride_f0 * 2],
                        strides=[stride_f0],
                        padding="SAME",
                        kernel_init=normal_init(0.01)
                    )
                )
                #noise_conv_norms.append(nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01)))
            else:
                noise_convs.append(
                    nn.Conv(features=self.hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=[1],
                           kernel_init=normal_init(0.01))
                )
                #noise_conv_norms.append(nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01)))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        resblocks = []#nn.ModuleList()
        #resblocks_norms=[]
        for i in range(len(ups)):
            ch = self.hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(self.hp.gen.resblock_kernel_sizes, self.hp.gen.resblock_dilation_sizes):
                resblocks.append(AMPBlock(ch, k, d,self.train))
                #resblocks_norms.append(nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01)))

        # post conv
        self.conv_post = nn.Conv(features=1, kernel_size=[7], strides=1, padding="SAME", use_bias=False,kernel_init=normal_init(0.01))
        # weight initialization
        self.ups = ups
        self.noise_convs = noise_convs
        self.resblocks = resblocks
        #self.noise_conv_norms = noise_conv_norms
        #self.resblocks_norms = resblocks_norms
        #self.norm1=nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01))
        #self.norm2=nn.BatchNorm(use_running_average=False, axis=-1,scale_init=normal_init(0.01))
        #self.ups_norm=ups_norm
        #self.ups.apply(init_weights)

    def __call__(self, spk, x, f0):
        rng = jax.random.PRNGKey(1234)
        # Perturbation
        x = x + jax.random.normal(rng,x.shape)  
        # adapter
        x = self.adapter(x, spk)
        # nsf
        f0 = f0[:, None]
        B, H, W = f0.shape
        
        f0 = jax.image.resize(f0, shape=(B, H, W * self.scale_factor), method='nearest').transpose(0,2,1)
        #f0 = self.f0_upsamp(f0).transpose(1, 2)
        
        har_source = self.m_source(f0)
        #har_source = har_source.transpose(0,2,1)
        x = x.transpose(0,2,1)
        x = self.conv_pre(x)
        #x = self.norm1(x)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, 0.1)
            # upsampling
            x = self.ups[i](x)
            #x = self.ups_norm[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
            #x_source = self.noise_conv_norms[i](x_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    #xs = self.resblocks_norms[i * self.num_kernels + j](xs)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    #xs = self.resblocks_norms[i * self.num_kernels + j](xs)
            x = xs / self.num_kernels

        # post conv
        x = nn.leaky_relu(x)
        x = self.conv_post(x)
        #x = self.norm2(x)
        x = x.transpose(0,2,1)
       
        x = nn.tanh(x) 
        return x

    # def remove_weight_norm(self):
    #     for l in self.ups:
    #         remove_weight_norm(l)
    #     for l in self.resblocks:
    #         l.remove_weight_norm()

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def pitch2source(self, f0):
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)  # [1,len,1]
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [1,1,len]
        return har_source

    def source2wav(self, audio):
        MAX_WAV_VALUE = 32768.0
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio.cpu().detach().numpy()

    def inference(self, spk, x, har_source):
        # adapter
        x = self.adapter(x, spk)
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

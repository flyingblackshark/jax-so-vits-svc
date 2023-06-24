# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from vits import commons
class DiscriminatorR(nn.Module):
    hp:tuple
    resolution:tuple
    def setup(self):
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope

        self.convs = [
            nn.Conv(features=32, kernel_size=[3, 9]),
            nn.Conv(features=32, kernel_size=[3, 9], strides=[1, 2]),
            nn.Conv(features=32, kernel_size=[3, 9], strides=[1, 2]),
            nn.Conv(features=32, kernel_size=[3, 9], strides=[1, 2]),
            nn.Conv(features=32, kernel_size=[3, 3]),
        ]

        self.conv_post = nn.Conv(features=1, kernel_size=[3, 3])
       
    def __call__(self, x,train=True):
        fmap = []
       
        x = self.spectrogram(x)
        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, self.hp.mpd.lReLU_slope)
            fmap.append(x)
        x = self.conv_post(x)

        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = jnp.pad(x, [(0,0),(0,0),(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2))], mode='reflect')
        x = jax.scipy.signal.stft(x, nfft=n_fft, noverlap=hop_length, nperseg=win_length) #[B, F, TT, 2]
        mag = jnp.clip(a=jnp.abs(x[2]),a_min=(1e-9))
        return mag


class MultiResolutionDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.resolutions = eval(self.hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(self.hp, resolution) for resolution in self.resolutions]
        

    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score)]

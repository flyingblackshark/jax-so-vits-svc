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
        self.norms = [nn.BatchNorm() for i in range(5)]
        self.conv_post = nn.Conv(features=1, kernel_size=[3, 3])
       
    def __call__(self, x,train=True):
        fmap = []
       
        x = self.spectrogram(x)
        x = jnp.expand_dims(x,1)
        for l,n in zip(self.convs,self.norms):
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = n(x.transpose(0,2,3,1),use_running_average=not train).transpose(0,3,1,2)
            x = nn.leaky_relu(x, self.hp.mpd.lReLU_slope)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = jnp.pad(x, [(0,0),(0,0),(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2))], mode='reflect')
        x = x.squeeze(1)
        x = jax.scipy.signal.stft(x,fs=32000, nfft=n_fft, noverlap=win_length-hop_length, nperseg=win_length) #[B, F, TT, 2]
        mag = jnp.abs(x[2])
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

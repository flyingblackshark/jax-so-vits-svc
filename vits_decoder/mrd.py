# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import jax
import jax.numpy as jnp
import flax.linen as nn
class DiscriminatorR(nn.Module):
    hp:tuple
    resolution:tuple
    def setup(self):
        #super(DiscriminatorR, self).__init__()

        #self.resolution = resolution
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope

       #norm_f = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = [
            nn.Conv(features=32, kernel_size=(3, 9), padding="same"),
            nn.Conv(features=32, kernel_size=(3, 9), strides=(1, 2), padding="same"),
            nn.Conv(features=32, kernel_size=(3, 9), strides=(1, 2), padding="same"),
            nn.Conv(features=32, kernel_size=(3, 9), strides=(1, 2), padding="same"),
            nn.Conv(features=32, kernel_size=(3, 3), padding="same"),
        ]
        self.conv_post = nn.Conv(features=1, kernel_size=(3, 3), padding="same")

    def __call__(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = jnp.expand_dims(x,0)
        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = jnp.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        #x = x.squeeze(1)
        x = jax.scipy.signal.stft(x, nfft=n_fft, noverlap=hop_length, nperseg=win_length) #[B, F, TT, 2]
        #mag = jnp.linalg.norm(x[2], ord=2, axis =-1) #[B, F, TT]
        mag = jnp.sqrt(jnp.real(x[2])**2+jnp.imag(x[2])**2)
        return mag


class MultiResolutionDiscriminator(nn.Module):
    hp:tuple
    def setup(self):
        #super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(self.hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(self.hp, resolution) for resolution in self.resolutions]
        

    def __call__(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]

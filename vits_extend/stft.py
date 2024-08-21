# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import audax
import jax.numpy as jnp
import flax.linen as nn
from librosa.filters import mel as librosa_mel_fn

class TacotronSTFT(nn.Module):

    def __init__(self, filter_length=512, hop_length=160, win_length=512,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=None):
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        self.mel_basis = jnp.asarray(mel)
    # def linear_spectrogram(self, y):
    #     spec = jax.scipy.signal.stft(y,nfft=self.n_fft, noverlap=self.win_size-self.hop_size, nperseg=self.win_size,return_onesided=True,padded=True,boundary=None)    
    #     hann_win = scipy.signal.get_window('hann',self.n_fft)
    #     scale = np.sqrt(1.0/hann_win.sum()**2)
    #     return jnp.abs(spec[2]/scale)

    def mel_spectrogram(self, y):
        hann_win = jnp.hanning(self.win_size)
        pad_size = (self.win_size-self.hop_size)//2
        wav = jnp.pad(wav, ((0,0),(pad_size, pad_size)),mode="reflect")
        spec = audax.core.stft.stft(wav,self.n_fft,self.hop_size,self.win_size,hann_win,onesided=True,center=False)
        spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
        spec = spec.transpose(0,2,1)
        spec = jnp.matmul(self.mel_basis, spec)
        spec = self.spectral_normalize_torch(spec)

        return spec

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return jnp.log(jnp.clip(x,min=clip_val) * C)

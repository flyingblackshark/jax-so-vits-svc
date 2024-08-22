import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from scipy.io.wavfile import read

MATPLOTLIB_FLAG = False

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    #is_torch = isinstance(f0, jax.Tensor)
    f0_mel = 1127 * jnp.log(1 + f0 / 700)
    #f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel = jnp.where(f0_mel>0,(f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,f0_mel)

    #f0_mel[f0_mel <= 1] = 1
    f0_mel = jnp.where(f0_mel<=1,1,f0_mel)
    #f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_mel = jnp.where(f0_mel > (f0_bin - 1),f0_bin - 1,f0_mel)
    f0_coarse = jnp.rint(f0_mel).astype(jnp.int32)
    # assert f0_coarse.max() <= 255 and f0_coarse.min(
    # ) >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse

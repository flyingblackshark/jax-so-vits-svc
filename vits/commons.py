import math
import numpy as np
from flax import linen as nn
import jax.numpy as jnp
import jax


def slice_pitch_segments(x, ids_str, segment_size=4):
    ret = jnp.zeros_like(x[:, :segment_size])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        #idx_end = idx_str + segment_size
        #ret[i] = x[i, idx_str:idx_end]
        ret = ret.at[i].set(jax.lax.dynamic_slice(x[i,:],[idx_str],[segment_size]))
    return ret


def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    rng=jax.random.PRNGKey(1234)
    ids_str = (jax.random.uniform(rng,[b]) * ids_str_max).astype(jnp.int32)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str


def rand_spec_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size
    rng=jax.random.PRNGKey(1234)
    ids_str = (jax.random.uniform(rng,[b]).to(device=x.device) * ids_str_max)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


# def init_weights(m, mean=0.0, std=0.01):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


# def kl_divergence(m_p, logs_p, m_q, logs_q):
#     """KL(P||Q)"""
#     kl = (logs_q - logs_p) - 0.5
#     kl += (
#         0.5 * (jnp.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * jnp.exp(-2.0 * logs_q)
#     )
#     return kl


def rand_gumbel(shape):
    rng=jax.random.PRNGKey(1234)
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = jax.random.uniform(rng,shape) * 0.99998 + 0.00001
    return -jnp.log(-jnp.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = jnp.zeros_like(x[:, :, :segment_size])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        #idx_end = idx_str + segment_size
        #ret[i] = x[i, :, idx_str:idx_end]
        ret = ret.at[i].set(jax.lax.dynamic_slice(x[i, :, :],(0,idx_str),(x.shape[1],segment_size)))
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    rng=jax.random.PRNGKey(1234)
    ids_str = (jax.random.uniform(rng,[b]) * ids_str_max)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = jnp.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * jnp.exp(
        jnp.arange(num_timescales) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], 0)
    signal = jnp.pad(signal, [0, 0, 0, channels % 2])
    signal = jnp.reshape(signal,[1, channels, length])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return jnp.concatenate([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = jnp.tril(jnp.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask



def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = nn.tanh(in_act[:, :n_channels_int, :])
    s_act = nn.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


# def convert_pad_shape(pad_shape):
#     l = pad_shape[::-1]
#     pad_shape = [item for sublist in l for item in sublist]
#     return pad_shape


# def shift_1d(x):
#     x = jnp.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
#     return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = jnp.arange(max_length)
    return jnp.expand_dims(x,0) < jnp.expand_dims(length,1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = jnp.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - jnp.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


# def clip_grad_value_(parameters, clip_value, norm_type=2):
#     # if isinstance(parameters, torch.Tensor):
#     #     parameters = [parameters]
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     norm_type = float(norm_type)
#     if clip_value is not None:
#         clip_value = float(clip_value)

#     total_norm = 0
#     for p in parameters:
#         param_norm = p.grad.data.norm(norm_type)
#         total_norm += param_norm.item() ** norm_type
#         if clip_value is not None:
#             p.grad.data.clamp_(min=-clip_value, max=clip_value)
#     total_norm = total_norm ** (1.0 / norm_type)
#     return total_norm
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial
import numpy as np
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax import lax

Array = Any

@jax.jit
def snake(x: Array, frequency: int = 1) -> Array:

    r"""Snake activation to learn periodic functions.

    Computes snake activation:

    $$
    \mathrm{snake}(x) = \mathrm{x} + \frac{1 - \cos(2 \cdot \mathrm{frequency} \cdot x)}{2 \cdot \mathrm{frequency}}.
    $$

    See [Neural Networks Fail to Learn Periodic Functions and How to Fix It](https://arxiv.org/abs/2006.08195).

    Usage:

    >>> x = tf.constant([-1.0, 0.0, 1.0])
    >>> tfa.activations.snake(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.29192656,  0.        ,  1.7080734 ], dtype=float32)>

    Args:
        x: A `Tensor`.
        frequency: A scalar, frequency of the periodic part.
    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    #x = tf.convert_to_tensor(x)
    #frequency = tf.cast(frequency, x.dtype)

    return x + (1 - jnp.cos(2 * frequency * x)) / (2 * frequency)
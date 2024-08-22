import math
import numpy as np
from flax import linen as nn
import jax.numpy as jnp
import jax


def slice_pitch_segments(x, ids_str, segment_size=4):
    ret = jnp.zeros_like(x[:, :segment_size])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        ret = ret.at[i].set(jax.lax.dynamic_slice(x,[i,idx_str],[1,segment_size]).squeeze(0))
    return ret


def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4,rng=None):
    b, d, t = x.shape
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (jax.random.uniform(rng,[b],maxval=ids_str_max)).astype(jnp.int32)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str


def slice_segments(x, ids_str, segment_size=4):
    ret = jnp.zeros_like(x[:, :, :segment_size])
    for i in range(x.shape[0]):
        idx_str = ids_str[i]
        ret = ret.at[i].set(jax.lax.dynamic_slice(x,[i,0,idx_str],[1,x.shape[1],segment_size]).squeeze(0))
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4,rng=None):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    uniform_key,rng = jax.random.split(rng,2)
    ids_str = (jax.random.uniform(uniform_key,[b]) * ids_str_max)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = jnp.arange(max_length,dtype=length.dtype)
    return jnp.expand_dims(x,0) < jnp.expand_dims(length,1)

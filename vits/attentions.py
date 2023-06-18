# import copy
# import math
import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
import jax.numpy as jnp
from flax import linen as nn
from vits import commons
from functools import partial
import flax
import jax
from jax.nn.initializers import normal as normal_init
class Encoder(nn.Module):
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int = 1
    p_dropout:float = 0.0
    window_size:int = 4
    #train:bool = True
    def setup(self):
        self.drop = nn.Dropout(self.p_dropout)
        attn_layers = []
        norm_layers_1 = []
        ffn_layers = []
        norm_layers_2 = []
        for i in range(self.n_layers):
            attn_layers.append(
               nn.attention.SelfAttention(
                    qkv_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                    num_heads=self.n_heads,
                    dropout_rate=self.p_dropout,
                    kernel_init=nn.initializers.xavier_normal()

                )
            )
            norm_layers_1.append(nn.LayerNorm(scale_init=normal_init(0.01)))
            ffn_layers.append(
                FFN(
                    self.hidden_channels,
                    self.filter_channels,
                    self.kernel_size,
                    p_dropout=self.p_dropout
                )
            )
            norm_layers_2.append(nn.LayerNorm(scale_init=normal_init(0.01)))
        self.attn_layers = attn_layers
        self.norm_layers_1 = norm_layers_1
        self.ffn_layers = ffn_layers
        self.norm_layers_2 = norm_layers_2
    #@nn.compact
    def __call__(self, x, x_mask,train=True):
        attn_mask = jnp.expand_dims(x_mask,2) * jnp.expand_dims(x_mask,-1)
        x = x * x_mask

        for i in range(self.n_layers):
            y = self.attn_layers[i](x.transpose(0,2,1),mask=attn_mask,deterministic=not train).transpose(0,2,1)
            y = self.drop(y.transpose(0,2,1),deterministic=not train).transpose(0,2,1)
            x = self.norm_layers_1[i]((x + y).transpose(0,2,1)).transpose(0,2,1)

            y = self.ffn_layers[i](x, x_mask,train=train)
            y = self.drop(y.transpose(0,2,1),deterministic=not train).transpose(0,2,1)
            x = self.norm_layers_2[i]((x + y).transpose(0,2,1)).transpose(0,2,1)

        x = x * x_mask
        return x


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         hidden_channels,
#         filter_channels,
#         n_heads,
#         n_layers,
#         kernel_size=1,
#         p_dropout=0.0,
#         proximal_bias=False,
#         proximal_init=True,
#         **kwargs
#     ):
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.filter_channels = filter_channels
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.kernel_size = kernel_size
#         self.p_dropout = p_dropout
#         self.proximal_bias = proximal_bias
#         self.proximal_init = proximal_init

#         self.drop = nn.Dropout(p_dropout)
#         self.self_attn_layers = []
#         self.norm_layers_0 = []
#         self.encdec_attn_layers = []
#         self.norm_layers_1 = []
#         self.ffn_layers = []
#         self.norm_layers_2 = []
#         for i in range(self.n_layers):
#             self.self_attn_layers.append(
#                 nn.attention.MultiHeadDotProductAttention(
#                     # hidden_channels,
#                     # hidden_channels,
#                     num_heads=n_heads,
#                     dropout_rate=p_dropout,
#                     # proximal_bias=proximal_bias,
#                     # proximal_init=proximal_init,
#                 )
#             )
#             self.norm_layers_0.append(LayerNorm(hidden_channels))
#             self.encdec_attn_layers.append(
#                 nn.attention.MultiHeadDotProductAttention(
#                    #hidden_channels, hidden_channels, 
#                     num_heads=n_heads, dropout_rate=p_dropout
#                 )
#             )
#             self.norm_layers_1.append(LayerNorm(hidden_channels))
#             self.ffn_layers.append(
#                 FFN(
#                     hidden_channels,
#                     hidden_channels,
#                     filter_channels,
#                     kernel_size,
#                     p_dropout=p_dropout,
#                     causal=True,
#                 )
#             )
#             self.norm_layers_2.append(LayerNorm(hidden_channels))

#     def __call__(self, x, x_mask, h, h_mask):
#         """
#         x: decoder input
#         h: encoder output
#         """
#         self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
#             device=x.device, dtype=x.dtype
#         )
#         encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
#         x = x * x_mask
#         for i in range(self.n_layers):
#             y = self.self_attn_layers[i](x, x, self_attn_mask)
#             y = self.drop(y)
#             x = self.norm_layers_0[i](x + y)

#             y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
#             y = self.drop(y)
#             x = self.norm_layers_1[i](x + y)

#             y = self.ffn_layers[i](x, x_mask)
#             y = self.drop(y)
#             x = self.norm_layers_2[i](x + y)
#         x = x * x_mask
#         return x


# class MultiHeadAttention(nn.Module):
#     def __init__(
#         self,
#         channels,
#         out_channels,
#         n_heads,
#         p_dropout=0.0,
#         window_size=None,
#         heads_share=True,
#         block_length=None,
#         proximal_bias=False,
#         proximal_init=False,
#     ):
#         super().__init__()
#         assert channels % n_heads == 0

#         self.channels = channels
#         self.out_channels = out_channels
#         self.n_heads = n_heads
#         self.p_dropout = p_dropout
#         self.window_size = window_size
#         self.heads_share = heads_share
#         self.block_length = block_length
#         self.proximal_bias = proximal_bias
#         self.proximal_init = proximal_init
#         self.attn = None

#         self.k_channels = channels // n_heads
#         self.conv_q = nn.Conv1d(channels, channels, 1)
#         self.conv_k = nn.Conv1d(channels, channels, 1)
#         self.conv_v = nn.Conv1d(channels, channels, 1)
#         self.conv_o = nn.Conv1d(channels, out_channels, 1)
#         self.drop = nn.Dropout(p_dropout)

#         if window_size is not None:
#             n_heads_rel = 1 if heads_share else n_heads
#             rel_stddev = self.k_channels**-0.5
#             self.emb_rel_k = nn.Parameter(
#                 torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
#                 * rel_stddev
#             )
#             self.emb_rel_v = nn.Parameter(
#                 torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
#                 * rel_stddev
#             )

#         nn.init.xavier_uniform_(self.conv_q.weight)
#         nn.init.xavier_uniform_(self.conv_k.weight)
#         nn.init.xavier_uniform_(self.conv_v.weight)
#         if proximal_init:
#             with torch.no_grad():
#                 self.conv_k.weight.copy_(self.conv_q.weight)
#                 self.conv_k.bias.copy_(self.conv_q.bias)

#     def forward(self, x, c, attn_mask=None):
#         q = self.conv_q(x)
#         k = self.conv_k(c)
#         v = self.conv_v(c)

#         x, self.attn = self.attention(q, k, v, mask=attn_mask)

#         x = self.conv_o(x)
#         return x

#     def attention(self, query, key, value, mask=None):
#         # reshape [b, d, t] -> [b, n_h, t, d_k]
#         b, d, t_s, t_t = (*key.size(), query.size(2))
#         query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
#         key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
#         value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

#         scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
#         if self.window_size is not None:
#             assert (
#                 t_s == t_t
#             ), "Relative attention is only available for self-attention."
#             key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
#             rel_logits = self._matmul_with_relative_keys(
#                 query / math.sqrt(self.k_channels), key_relative_embeddings
#             )
#             scores_local = self._relative_position_to_absolute_position(rel_logits)
#             scores = scores + scores_local
#         if self.proximal_bias:
#             assert t_s == t_t, "Proximal bias is only available for self-attention."
#             scores = scores + self._attention_bias_proximal(t_s).to(
#                 device=scores.device, dtype=scores.dtype
#             )
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e4)
#             if self.block_length is not None:
#                 assert (
#                     t_s == t_t
#                 ), "Local attention is only available for self-attention."
#                 block_mask = (
#                     torch.ones_like(scores)
#                     .triu(-self.block_length)
#                     .tril(self.block_length)
#                 )
#                 scores = scores.masked_fill(block_mask == 0, -1e4)
#         p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
#         p_attn = self.drop(p_attn)
#         output = torch.matmul(p_attn, value)
#         if self.window_size is not None:
#             relative_weights = self._absolute_position_to_relative_position(p_attn)
#             value_relative_embeddings = self._get_relative_embeddings(
#                 self.emb_rel_v, t_s
#             )
#             output = output + self._matmul_with_relative_values(
#                 relative_weights, value_relative_embeddings
#             )
#         output = (
#             output.transpose(2, 3).contiguous().view(b, d, t_t)
#         )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
#         return output, p_attn

#     def _matmul_with_relative_values(self, x, y):
#         """
#         x: [b, h, l, m]
#         y: [h or 1, m, d]
#         ret: [b, h, l, d]
#         """
#         ret = torch.matmul(x, y.unsqueeze(0))
#         return ret

#     def _matmul_with_relative_keys(self, x, y):
#         """
#         x: [b, h, l, d]
#         y: [h or 1, m, d]
#         ret: [b, h, l, m]
#         """
#         ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
#         return ret

#     def _get_relative_embeddings(self, relative_embeddings, length):
#         max_relative_position = 2 * self.window_size + 1
#         # Pad first before slice to avoid using cond ops.
#         pad_length = max(length - (self.window_size + 1), 0)
#         slice_start_position = max((self.window_size + 1) - length, 0)
#         slice_end_position = slice_start_position + 2 * length - 1
#         if pad_length > 0:
#             padded_relative_embeddings = F.pad(
#                 relative_embeddings,
#                 commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
#             )
#         else:
#             padded_relative_embeddings = relative_embeddings
#         used_relative_embeddings = padded_relative_embeddings[
#             :, slice_start_position:slice_end_position
#         ]
#         return used_relative_embeddings

#     def _relative_position_to_absolute_position(self, x):
#         """
#         x: [b, h, l, 2*l-1]
#         ret: [b, h, l, l]
#         """
#         batch, heads, length, _ = x.size()
#         # Concat columns of pad to shift from relative to absolute indexing.
#         x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

#         # Concat extra elements so to add up to shape (len+1, 2*len-1).
#         x_flat = x.view([batch, heads, length * 2 * length])
#         x_flat = F.pad(
#             x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
#         )

#         # Reshape and slice out the padded elements.
#         x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
#             :, :, :length, length - 1 :
#         ]
#         return x_final

#     def _absolute_position_to_relative_position(self, x):
#         """
#         x: [b, h, l, l]
#         ret: [b, h, l, 2*l-1]
#         """
#         batch, heads, length, _ = x.size()
#         # padd along column
#         x = F.pad(
#             x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
#         )
#         x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
#         # add 0's in the beginning that will skew the elements after reshape
#         x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
#         x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
#         return x_final

#     def _attention_bias_proximal(self, length):
#         """Bias for self-attention to encourage attention to close positions.
#         Args:
#           length: an integer scalar.
#         Returns:
#           a Tensor with shape [1, 1, length, length]
#         """
#         r = torch.arange(length, dtype=torch.float32)
#         diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
#         return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

from jax.nn.initializers import normal as normal_init
class FFN(nn.Module):
    out_channels:int
    filter_channels:int
    kernel_size:int
    p_dropout:float=0.0
    activation:str=None
    causal:bool=False

    def setup(self):
        if self.causal:
            self.padding = "CAUSAL"
        else:
            self.padding = "SAME"
        self.conv_1 = nn.Conv(self.filter_channels, [self.kernel_size],padding=self.padding)
        self.conv_2 = nn.Conv(self.out_channels, [self.kernel_size],padding=self.padding)
        self.drop = nn.Dropout(self.p_dropout)


    def __call__(self, x, x_mask,train=True):
        x = self.conv_1((x * x_mask).transpose(0,2,1)).transpose(0,2,1)
        if self.activation == "gelu":
            x = nn.gelu(x)
        else:
            x = nn.relu(x)
        x = self.drop(x.transpose(0,2,1),deterministic=not train).transpose(0,2,1)
        x = self.conv_2((x * x_mask).transpose(0,2,1)).transpose(0,2,1)
        return x * x_mask

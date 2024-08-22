import jax.numpy as jnp
from flax import linen as nn

class Encoder(nn.Module):
    hidden_channels:int
    filter_channels:int
    n_heads:int
    n_layers:int
    kernel_size:int = 1
    p_dropout:float = 0.0
    window_size:int = 4

    def setup(self):
        self.drop = nn.Dropout(self.p_dropout)
        attn_layers = []
        norm_layers_1 = []
        ffn_layers = []
        norm_layers_2 = []
        for i in range(self.n_layers):
            attn_layers.append(
                nn.SelfAttention(self.n_heads,qkv_features=self.hidden_channels,out_features=self.hidden_channels,dropout_rate=self.p_dropout)
            )
            norm_layers_1.append(nn.LayerNorm())
            ffn_layers.append(
                FFN(
                    self.hidden_channels,
                    self.filter_channels,
                    self.kernel_size,
                    p_dropout=self.p_dropout
                )
            )
            norm_layers_2.append(nn.LayerNorm())
        self.attn_layers = attn_layers
        self.norm_layers_1 = norm_layers_1
        self.ffn_layers = ffn_layers
        self.norm_layers_2 = norm_layers_2

    def __call__(self, x, x_mask,train=True):
        attn_mask = jnp.expand_dims(x_mask,2) * jnp.expand_dims(x_mask,-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x.transpose(0,2,1), mask=attn_mask,deterministic=not train).transpose(0,2,1)
            y = self.drop(y,deterministic=not train)
            x = self.norm_layers_1[i]((x + y).transpose(0,2,1)).transpose(0,2,1)

            y = self.ffn_layers[i](x, x_mask,train=train)
            y = self.drop(y,deterministic=not train)
            x = self.norm_layers_2[i]((x + y).transpose(0,2,1)).transpose(0,2,1)
        x = x * x_mask
        return x


class FFN(nn.Module):
    out_channels:int
    filter_channels:int
    kernel_size:int
    p_dropout:float=0.0

    def setup(self):
        self.conv_1 = nn.Conv(self.filter_channels, [self.kernel_size])
        self.conv_2 = nn.Conv(self.out_channels, [self.kernel_size])
        self.drop = nn.Dropout(self.p_dropout)

    def __call__(self, x, x_mask,train=True):
        x = self.conv_1((x*x_mask).transpose(0,2,1)).transpose(0,2,1)
        x = nn.gelu(x)
        x = self.drop(x,deterministic=not train)
        x = self.conv_2((x*x_mask).transpose(0,2,1)).transpose(0,2,1)
        return x*x_mask

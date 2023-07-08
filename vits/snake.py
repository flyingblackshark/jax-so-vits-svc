import jax.numpy as jnp
import flax
from flax import linen as nn
from jax.nn.initializers import normal as normal_init
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
from jax import custom_vjp
class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''

    # def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
    #     '''
    #     Initialization.
    #     INPUT:
    #         - in_features: shape of the input
    #         - alpha - trainable parameter that controls frequency
    #         - beta - trainable parameter that controls magnitude
    #         alpha is initialized to 1 by default, higher values = higher-frequency.
    #         beta is initialized to 1 by default, higher values = higher-magnitude.
    #         alpha will be trained along with the rest of your model.
    #     '''
    #     #super(SnakeBeta, self).__init__()
    #     self.in_features = in_features
    #     # initialize alpha
    #     self.alpha_logscale = alpha_logscale
    #     if self.alpha_logscale:  # log scale alphas initialized to zeros
    #         self.alpha = Parameter(torch.zeros(in_features) * alpha)
    #         self.beta = Parameter(torch.zeros(in_features) * alpha)
    #     else:  # linear scale alphas initialized to ones
    #         self.alpha = Parameter(torch.ones(in_features) * alpha)
    #         self.beta = Parameter(torch.ones(in_features) * alpha)
    #     self.alpha.requires_grad = alpha_trainable
    #     self.beta.requires_grad = alpha_trainable
    #     self.no_div_by_zero = 0.000000001
    @nn.compact
    def __call__(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta = x + 1/b * sin^2 (xa)
        '''
        no_div_by_zero = 0.000000001
        alpha = self.param("alpha",nn.initializers.zeros,x.shape[-2])
        beta = self.param("beta",nn.initializers.zeros,x.shape[-2])

        alpha = jnp.expand_dims(jnp.expand_dims(alpha,0),-1)  # line up with x to [B, C, T]
        beta = jnp.expand_dims(jnp.expand_dims(beta,0),-1)
        alpha = jnp.exp(alpha)
        beta = jnp.exp(beta)
        x = x + (1.0 / (beta + no_div_by_zero)) * jnp.square(jnp.sin(x * alpha))
        return x

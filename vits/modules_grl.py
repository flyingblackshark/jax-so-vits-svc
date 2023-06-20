import jax.numpy as jnp
from jax import custom_jvp

import flax
from flax import linen as nn
from jax.nn.initializers import normal as normal_init
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
@jax.custom_jvp
def gradient_reversal(x):
  return x

@gradient_reversal.defjvp
def f_jvp(primals,tangents):
    x = primals
    primal_out = gradient_reversal(x)
    tangent_out = -1
    return primal_out, tangent_out


# class GradientReversal(nn.Module):
#     ''' Gradient Reversal Layer
#             Y. Ganin, V. Lempitsky,
#             "Unsupervised Domain Adaptation by Backpropagation",
#             in ICML, 2015.
#         Forward pass is the identity function
#         In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
#     '''

#     # def __init__(self, lambda_reversal=1):
#     #     super(GradientReversal, self).__init__()
#     #     self.lambda_ = lambda_reversal

#     def __call__(self, x):
#         #return GradientReversalFunction.apply(x, self.lambda_)
#         return gradient_reversal(x)


class SpeakerClassifier(nn.Module):
    embed_dim:int
    spk_dim:int
    def setup(self):
       # super(SpeakerClassifier, self).__init__()
        self.classifier = nn.Sequential([
            gradient_reversal,
            nn.Conv( self.embed_dim, kernel_size=[5]),
            nn.relu,
            nn.Conv( self.embed_dim, kernel_size=[5]),
            nn.relu,
            nn.Conv( self.spk_dim, kernel_size=[5])]
        )

    def __call__(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim, len)
        '''
        # pass through classifier
        # temp = jax.grad(self.classifier)(x)
        # jax.debug.print("{}",temp)
        outputs = self.classifier(x.transpose(0,2,1)).transpose(0,2,1) # (B, nb_speakers)
        outputs = jnp.mean(outputs, axis=-1)
        return outputs

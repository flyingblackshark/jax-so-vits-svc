import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from vits import commons
class ScaleDiscriminator(nn.Module):
    def setup(self):

        self.convs = [
            nn.Conv(16, [15], 1),
            nn.Conv(64, [41], 4, feature_group_count =4),
            nn.Conv( 256, [41], 4, feature_group_count =16),
            nn.Conv( 1024, [41], 4, feature_group_count =64),
            nn.Conv( 1024, [41], 4, feature_group_count =256),
            nn.Conv( 1024, [5], 1),
        ]
        self.norms = [nn.BatchNorm(axis=-1,scale_init=normal_init(0.01)) for i in range(6)]
        self.conv_post = nn.Conv( 1, [3], 1)

    def __call__(self, x,train=True):
        fmap = []
        for l,n in zip(self.convs,self.norms):
            x = l(x.transpose(0,2,1)).transpose(0,2,1)
            x = commons.snake(x)
            x = n(x.transpose(0,2,1),use_running_average=not train).transpose(0,2,1)
            #x = nn.leaky_relu(x, 0.1)
        
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        fmap.append(x)
        x = jnp.reshape(x,[x.shape[0],-1])
        return [(fmap, x)]

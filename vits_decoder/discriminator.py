# import torch
# import torch.nn as nn
from flax import linen as nn
import jax.numpy as jnp
import jax
from omegaconf import OmegaConf

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
#from .msd import ScaleDiscriminator

class Discriminator(nn.Module):
    hp:tuple
    def setup(self):
        #super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(self.hp)
        self.MPD = MultiPeriodDiscriminator(self.hp)
        #self.MSD = ScaleDiscriminator()


    def __call__(self, x,train=True):
        r = self.MRD(x,train=train)
        p = self.MPD(x,train=train)
        #s = self.MSD(x,train=train)
        return r + p 

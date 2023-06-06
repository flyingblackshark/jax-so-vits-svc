# import torch
# import torch.nn as nn
from flax import linen as nn
import jax.numpy as jnp
import jax
from omegaconf import OmegaConf

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator


class Discriminator(nn.Module):
    hp:tuple
    def setup(self):
        #super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(self.hp)
        self.MPD = MultiPeriodDiscriminator(self.hp)


    def __call__(self, x):
        r = self.MRD(x)
        p = self.MPD(x)

        return r + p


# if __name__ == '__main__':
#     hp = OmegaConf.load('../config/base.yaml')
#     model = Discriminator(hp)

#     x = jax.random.no(3, 1, 16384)
#     print(x.shape)

#     output = model(x)
#     for features, score in output:
#         for feat in features:
#             print(feat.shape)
#         print(score.shape)

#     pytorch_total_params = sum(p.numel()
#                                for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)

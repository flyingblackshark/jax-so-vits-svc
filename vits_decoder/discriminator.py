from flax import linen as nn
from .mpd import MultiPeriodDiscriminator
from .msd import ScaleDiscriminator

class Discriminator(nn.Module):
    hp:tuple
    def setup(self):
        self.MPD = MultiPeriodDiscriminator(self.hp)
        self.MSD = ScaleDiscriminator()
    def __call__(self, x,train=True):
        p = self.MPD(x,train=train)
        s = self.MSD(x,train=train)
        return p + s

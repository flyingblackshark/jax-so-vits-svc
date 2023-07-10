import jax.numpy as jnp
from flax import linen as nn

class SpeakerClassifier(nn.Module):
    embed_dim:int
    spk_dim:int
    def setup(self):
        self.classifier = nn.Sequential([
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
        outputs = self.classifier(x.transpose(0,2,1)).transpose(0,2,1) # (B, nb_speakers)
        outputs = jnp.mean(outputs, axis=-1)
        return outputs

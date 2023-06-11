import tqdm
# import torch
# import torch.nn.functional as F
import flax
import jax
import optax
import numpy as np
from flax import linen as nn
import jax.numpy as jnp
from vits_extend.stft import TacotronSTFT
from vits_extend.writer import MyWriter
from vits.models import SynthesizerTrn
def validate(hp, args, generator, discriminator, valloader, stft, writer, step):
    # generator.eval()
    # discriminator.eval()
    # torch.backends.cudnn.benchmark = False
    model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp)
    
    stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax)
    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (ppg, ppg_l, pit, spk, spec, spec_l, audio, audio_l) in enumerate(loader):
        # ppg = ppg.to(device)
        # pit = pit.to(device)
        # spk = spk.to(device)
        # ppg_l = ppg_l.to(device)
        # audio = audio.to(device)

        # if hasattr(generator, 'module'):
        #     fake_audio = generator.module.infer(ppg, pit, spk, ppg_l)[
        #         :, :, :audio.size(2)]
        # else:
        fake_audio = model.apply({'params': generator.params}, ppg, pit, spk, ppg_l,method=SynthesizerTrn.infer)
        #fake_audio = generator.infer(ppg, pit, spk, ppg_l)[:, :, :audio.size(2)]

        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = stft.mel_spectrogram(audio.squeeze(1))

        mel_loss += jnp.mean(optax.l2_loss(mel_fake, mel_real)) 

        if idx < hp.log.num_audio:
            spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
            spec_real = stft.linear_spectrogram(audio.squeeze(1))

            audio = audio[0][0]
            fake_audio = fake_audio[0][0]
            spec_fake = spec_fake[0]
            spec_real = spec_real[0]
            writer.log_fig_audio(
                audio, fake_audio, spec_fake, spec_real, idx, step)

    mel_loss = mel_loss / len(valloader.dataset)

    writer.log_validation(mel_loss, generator, discriminator, step)

    #torch.backends.cudnn.benchmark = True

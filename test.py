import jax
from omegaconf import OmegaConf
from vits_extend.dataloader import create_dataloader_train
hp = OmegaConf.load("configs/base.yaml")
trainloader = create_dataloader_train(hp)
(fake_ppg,fake_ppg_l,fake_vec,fake_pit,fake_spk,fake_spec,fake_spec_l,fake_audio,wav_l) = next(iter(trainloader))
print()
import os
import time
import logging
import math
import tqdm
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributed import init_process_group
# from torch.nn.parallel import DistributedDataParallel
import itertools
import traceback
import flax
import jax
import optax
import numpy as np
from flax import linen as nn
from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
#from vits_extend.validation import validate
from vits_decoder.discriminator import Discriminator
from vits.models import SynthesizerTrn
from vits import commons
from vits.losses import kl_loss
#from vits.commons import clip_grad_value_
import jax.numpy as jnp

from functools import partial
from typing import Any, Tuple
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
import torch
PRNGKey = jnp.ndarray

# class TrainState(train_state.TrainState):
#     batch_stats: Any

def train(rank, args, chkpt_path, hp, hp_str):
    num_devices = jax.device_count()
    #torch.multiprocessing.set_start_method('spawn')
    #@partial(jax.pmap, static_broadcasted_argnums=(1))
    def create_generator_state(rng, model_cls): 
        r"""Create the training state given a model class. """ 
        model = model_cls(spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp)
        # tx = optax.chain(
        # optax.clip_by_global_norm(1),
        # optax.scale_by_adam(b1=hp.train.betas[0],b2=hp.train.betas[1], eps=hp.train.eps),
        # optax.scale(-hp.train.learning_rate))
        tx = optax.adamw(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1], eps=hp.train.eps)
        fake_ppg = jnp.ones((hp.train.batch_size,400,1280))
        fake_pit = jnp.ones((hp.train.batch_size,400))
        fake_spec = jnp.ones((hp.train.batch_size,513,400))
        fake_spk = jnp.ones((hp.train.batch_size,256))
        fake_spec_l = jnp.asarray(np.asarray(400))
        fake_ppg_l = jnp.asarray(np.asarray(400))

        variables = model.init(rng, ppg=fake_ppg, pit=fake_pit, spec=fake_spec, spk=fake_spk, ppg_l=fake_ppg_l, spec_l=fake_spec_l)

        state = TrainState.create(apply_fn=model.apply, tx=tx, 
            params=variables['params'])#,batch_stats=variables['batch_stats'])
        
        return state
    #@partial(jax.pmap, static_broadcasted_argnums=(1))
    def create_discriminator_state(rng, model_cls): 
        r"""Create the training state given a model class. """ 
        model = model_cls(hp=hp)
        fake_audio = jnp.ones((hp.train.batch_size,1,8000))
        # tx = optax.chain(
        # optax.clip_by_global_norm(1),
        # optax.scale_by_adam(b1=hp.train.betas[0],b2=hp.train.betas[1], eps=hp.train.eps),
        # optax.scale(-hp.train.learning_rate))
        tx = optax.adamw(learning_rate=hp.train.learning_rate, b1=hp.train.betas[0],b2=hp.train.betas[1], eps=hp.train.eps)
        variables = model.init(rng, fake_audio)
       
        state = TrainState.create(apply_fn=model.apply, tx=tx, 
            params=variables['params'])#, batch_stats=variables['batch_stats'])
        
        return state
    #@partial(jax.pmap, axis_name='num_devices')
    def combine_step(generator_state: TrainState,
                       discriminator_state: TrainState,
                       ppg : jnp.ndarray  , pit : jnp.ndarray, spec : jnp.ndarray, spk : jnp.ndarray, ppg_l : jnp.ndarray ,spec_l:jnp.ndarray ,audio_e:jnp.ndarray):
                      # key: PRNGKey):
      

        def loss_fn(params):
            stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax)
            stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
            (fake_audio, ids_slice, z_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r)),mutables = generator_state.apply_fn(
                {'params': params},#,'batch_stats': generator_state.batch_stats},     
                ppg, pit, spec, spk, ppg_l, spec_l)#, mutable=['batch_stats'])
            audio = commons.slice_segments(audio_e, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            #temp = jax.grad(stft.mel_spectrogram(fake_audio.squeeze(1)))
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            mel_loss = jnp.mean(optax.l2_loss(mel_fake, mel_real)) * hp.train.c_mel
          
            #Multi-Resolution STFT Loss
            
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

            # Generator Loss
            #disc_fake = model_d(fake_audio)
           
            disc_fake,_ = discriminator_state.apply_fn(
            {'params': discriminator_state.params}, #,'batch_stats': discriminator_state.batch_stats},
            fake_audio)#, mutable=['batch_stats'])
            score_loss = 0.0
            for (_, score_fake) in disc_fake:
                score_loss += jnp.mean((score_fake - 1.0)**2)
            score_loss = score_loss / len(disc_fake)

            # Feature Loss
            # disc_real = model_d(audio)
            disc_real,_ = discriminator_state.apply_fn(
            {'params': discriminator_state.params},#,'batch_stats': discriminator_state.batch_stats},
            audio)#, mutable=['batch_stats'])

            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += jnp.mean(jnp.abs(fake - real))
            feat_loss = feat_loss / len(disc_fake)
            feat_loss = feat_loss * 2

            # Kl Loss
            loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
            loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl
            loss_kl_f = jnp.mean(loss_kl_f)
            loss_kl_r = jnp.mean(loss_kl_r)
            # Loss
            loss_g = mel_loss +score_loss +  feat_loss + stft_loss+ loss_kl_f + loss_kl_r * 0.5# + spk_loss * 0.5

            return loss_g, (mutables,fake_audio,audio,mel_loss,stft_loss,loss_kl_f,loss_kl_r,score_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss,(mutables,fake_audio_g,audio_g,mel_loss,stft_loss,loss_kl_f,loss_kl_r,score_loss)), grads = grad_fn(generator_state.params)

        # Average across the devices.
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        loss_g = jax.lax.pmean(loss, axis_name='num_devices')
        loss_m = jax.lax.pmean(mel_loss, axis_name='num_devices')
        loss_s = jax.lax.pmean(stft_loss, axis_name='num_devices')
        loss_k = jax.lax.pmean(loss_kl_f, axis_name='num_devices')
        loss_r = jax.lax.pmean(loss_kl_r, axis_name='num_devices')

        new_generator_state = generator_state.apply_gradients(
            grads=grads)#, batch_stats=mutables['batch_stats'])
        
        def loss_fn(params):
            # (fake_audio, ids_slice, z_mask, \
            # (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r)),mutables = generator_state.apply_fn(
            #     {'params': generator_state.params, 'batch_stats': generator_state.batch_stats},
            #     ppg, pit, spec, spk, ppg_l, spec_l, mutable=['batch_stats'])
            # audio = commons.slice_segments(audio_e, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            disc_fake,mutables  = discriminator_state.apply_fn(
                {'params': params},#,'batch_stats': discriminator_state.batch_stats},    
             fake_audio_g)#, mutable=['batch_stats'])
            disc_real,mutables  = discriminator_state.apply_fn(
                {'params': params},#,'batch_stats':  mutables['batch_stats']},
                audio_g)#, mutable=['batch_stats'])
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += jnp.mean((score_real - 1.0)**2)
                loss_d += jnp.mean((score_fake)**2)
            loss_d = loss_d / len(disc_fake)
          
            return loss_d,mutables 
        
        # Generate data with the Generator, critique it with the Discriminator.
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        #optax.clip_by_global_norm(None)
        (loss,mutables), grads = grad_fn(discriminator_state.params)

        # Average cross the devices.
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        loss_d = jax.lax.pmean(loss, axis_name='num_devices')

        # Update the discriminator through gradient descent.
        new_discriminator_state = discriminator_state.apply_gradients(
        grads=grads)#, batch_stats=mutables['batch_stats'])
        return new_generator_state,new_discriminator_state,loss_g,loss_d,loss_m,loss_s,loss_k,loss_r,score_loss
    #@jax.pmap
    def validate(generator):
        # generator.eval()
        # discriminator.eval()
        # torch.backends.cudnn.benchmark = False
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
            segment_size=hp.data.segment_size // hp.data.hop_length,
            hp=hp,train=False)
        
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
            # ppg = shard(ppg)
            # pit = shard(pit)
            # spk = shard(spk)
            # ppg_l = shard(ppg_l)
            # audio = shard(audio)

            # if hasattr(generator, 'module'):
            #     fake_audio = generator.module.infer(ppg, pit, spk, ppg_l)[
            #         :, :, :audio.size(2)]
            # else:
            audio = audio[:,:,:32000]
            fake_audio = model.apply({'params': generator.params}, ppg, pit, spk, ppg_l,method=SynthesizerTrn.infer, mutable=False)
            #fake_audio = generator.infer(ppg, pit, spk, ppg_l)[:, :, :audio.size(2)]
            #jax.debug.print("{}",fake_audio.shape)
            #jax.debug.print("{}",audio.shape)
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))

            mel_loss += float(jnp.mean(optax.l2_loss(mel_fake, mel_real)))

            if idx < hp.log.num_audio:
                spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
                spec_real = stft.linear_spectrogram(audio.squeeze(1))

                audio = audio[0][0]
                fake_audio = fake_audio[0][0]
                spec_fake = spec_fake[0]
                spec_real = spec_real[0]
                fake_audio = np.asarray(fake_audio)
                audio = np.asarray(audio)
                writer.log_fig_audio(
                    audio, fake_audio, spec_fake, spec_real, idx, step)

        mel_loss = mel_loss / len(valloader.dataset)

        writer.log_validation(mel_loss, step)

    key = jax.random.PRNGKey(seed=hp.train.seed)
    key_generator, key_discriminator, key = jax.random.split(key, 3)
    # key_generator = shard_prng_key(key_generator)
    # key_discriminator = shard_prng_key(key_discriminator)

    discriminator_state = create_discriminator_state(key_discriminator, Discriminator)
    
    generator_state = create_generator_state(key_generator, SynthesizerTrn)

    init_epoch = 1
    step = 0
   


    if rank == 0:
        pth_dir = os.path.join(hp.log.pth_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pth_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        valloader = create_dataloader_eval(hp)
    


    trainloader = create_dataloader_train(hp, args.num_gpus, rank)

    for epoch in range(init_epoch, hp.train.epochs):

        #trainloader.batch_sampler.set_epoch(epoch)
        if rank == 0 and epoch % hp.log.eval_interval == 0:
            validate(generator_state)
        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader
        #data_gen = iter(loader)
        for ppg, ppg_l, pit, spk, spec, spec_l, audio, audio_l in loader:

            # ppg = shard(ppg)
            # ppg_l = shard(ppg_l)
            # pit = shard(pit)
            # spk = shard(spk)
            # spec = shard(spec)
            # spec_l = shard(spec_l)
            # audio = shard(audio)
            # audio_l = shard(audio_l)
            generator_state,discriminator_state,loss_g,loss_d,loss_m,loss_s,loss_k,loss_r,score_loss=combine_step(generator_state, discriminator_state,ppg=ppg,pit=pit, spk=spk, spec=spec,ppg_l=ppg_l,spec_l=spec_l,audio_e=audio)
            # generator_state, generator_loss,fake_audio_g,audio_g= generator_step(generator_state, discriminator_state,ppg=ppg,pit=pit, spk=spk, spec=spec,ppg_l=ppg_l,spec_l=spec_l,audio_e=audio,key=key_generator)
            # fake_audio_g = shard(fake_audio_g)
            # audio_g = shard(audio_g)
            # #print("Working!2")
            # discriminator_state, discriminator_loss = discriminator_step(generator_state, discriminator_state,audio_g=audio_g,fake_audio_g=fake_audio_g,key=key_discriminator)
            #print("Working!3")


            # discriminator



            step += 1
            # logging
            # loss_g = np.mean(np.asarray(loss_g))
            # loss_d = np.mean(np.asarray(loss_d))
            # loss_s = np.mean(np.asarray(loss_s))
            # loss_m = np.mean(np.asarray(loss_m))
            # loss_k = np.mean(np.asarray(loss_k))
            # loss_r = np.mean(np.asarray(loss_r))
            loss_g,loss_d,loss_s,loss_m,loss_k,loss_r,score_loss = \
            jax.device_get([loss_g[0], loss_d[0],loss_s[0],loss_m[0],loss_k[0],loss_r[0],score_loss[0]])
            if rank == 0 and step % hp.log.info_interval == 0:
                #metrics = jax.device_get([loss_g[0], loss_d[1]])
                # print("g %.04f d %.04f  | step %d" % (
                #      loss_g,  loss_d,  step))
                #print(metrics)
                # print(np.mean(loss_d))
                # print(np.mean(loss_g))
                writer.log_training(
                    loss_g, loss_d, loss_m, loss_s, loss_k, loss_r, score_loss,step)
                logger.info("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f  | step %d" % (
                    loss_g, loss_m, loss_s, loss_d, loss_k, loss_r, step))


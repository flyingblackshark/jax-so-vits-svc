import os
import time
import logging
import math
import tqdm
import itertools
import traceback
import flax
import jax
import optax
import numpy as np
import orbax
from flax import linen as nn
from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.writer import MyWriter
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from vits_decoder.discriminator import Discriminator
from vits.models import SynthesizerTrn
from vits import commons
from vits.losses import kl_loss
import jax.numpy as jnp
import orbax.checkpoint
from functools import partial
from typing import Any, Tuple
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
import torch
from flax.training import orbax_utils
from vits.losses import l1_loss
PRNGKey = jnp.ndarray

# class TrainState(train_state.TrainState):
#     batch_stats: Any

def train(rank, args, chkpt_path, hp, hp_str):
    num_devices = jax.device_count()
    total_steps = 180000
    @partial(jax.pmap, static_broadcasted_argnums=(1))
    def create_generator_state(rng, model_cls): 
        r"""Create the training state given a model class. """ 
        model = model_cls(spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp)
        
        exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=total_steps,
                                                      decay_rate=hp.train.lr_decay, transition_begin=int(total_steps*0.25),
                                                      staircase=False)
        tx = optax.adamw(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1], eps=hp.train.eps)
        # fake_ppg = jnp.ones((hp.train.batch_size,400,1280))
        # fake_pit = jnp.ones((hp.train.batch_size,400))
        # fake_spec = jnp.ones((hp.train.batch_size,513,400))
        # fake_spk = jnp.ones((hp.train.batch_size,256))
        # fake_spec_l = jnp.asarray(np.asarray([400 for i in range(hp.train.batch_size)]))
        # fake_ppg_l = jnp.asarray(np.asarray([400 for i in range(hp.train.batch_size)]))
        (fake_ppg,fake_ppg_l,fake_pit,fake_spk,fake_spec,fake_spec_l,wav,wav_l) = next(iter(trainloader))
        variables = model.init(rng, ppg=fake_ppg, pit=fake_pit, spec=fake_spec, spk=fake_spk, ppg_l=fake_ppg_l, spec_l=fake_spec_l,train=False)

        state = TrainState.create(apply_fn=model.apply, tx=tx, 
            params=variables['params'])#,batch_stats=variables['batch_stats'])
        
        return state
    @partial(jax.pmap, static_broadcasted_argnums=(1))
    def create_discriminator_state(rng, model_cls): 
        r"""Create the training state given a model class. """ 
        model = model_cls(hp=hp)
        #fake_audio = jnp.ones((hp.train.batch_size,1,8000))
        (fake_ppg,fake_ppg_l,fake_pit,fake_spk,fake_spec,fake_spec_l,fake_audio,wav_l) = next(iter(trainloader))
        fake_audio = fake_audio[:,:,:8000]
        exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=total_steps,
                                                      decay_rate=hp.train.lr_decay, transition_begin=int(total_steps*0.25),
                                                      staircase=False)
        tx = optax.adamw(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1], eps=hp.train.eps)
        variables = model.init(rng, fake_audio)

        state = TrainState.create(apply_fn=model.apply, tx=tx, 
            params=variables['params'])#, batch_stats=variables['batch_stats'])
        
        return state
    @partial(jax.pmap, axis_name='num_devices')
    def combine_step(generator_state: TrainState,
                       discriminator_state: TrainState,
                       ppg : jnp.ndarray  , pit : jnp.ndarray, spec : jnp.ndarray, spk : jnp.ndarray, ppg_l : jnp.ndarray ,spec_l:jnp.ndarray ,audio_e:jnp.ndarray
                      ):
      

        def loss_fn(params):
            stft = TacotronSTFT(filter_length=hp.data.filter_length,
                    hop_length=hp.data.hop_length,
                    win_length=hp.data.win_length,
                    n_mel_channels=hp.data.mel_channels,
                    sampling_rate=hp.data.sampling_rate,
                    mel_fmin=hp.data.mel_fmin,
                    mel_fmax=hp.data.mel_fmax)
            stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
            fake_audio, ids_slice, z_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r) = generator_state.apply_fn(
                {'params': params},#,'batch_stats': generator_state.batch_stats},     
                ppg, pit, spec, spk, ppg_l, spec_l,train=True, rngs={'dropout': jax.random.PRNGKey(1234)})#,mutable=['batch_stats'])
            audio = commons.slice_segments(audio_e, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            
           # mel_loss = jnp.mean(optax.l2_loss(mel_fake, mel_real)) * hp.train.c_mel
            mel_loss = l1_loss(mel_fake, mel_real) * hp.train.c_mel
            #spkc_criterion = 
            #spk_loss = optax.cosine_similarity(spk, spk_preds).mean()
            #Multi-Resolution STFT Loss
            
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

            # Generator Loss 
            disc_fake = discriminator_state.apply_fn(
            {'params': discriminator_state.params},#,'batch_stats': discriminator_state.batch_stats},
            fake_audio)#, mutable=['batch_stats'])
            score_loss = 0.0
            for (_, score_fake) in disc_fake:
                score_loss += jnp.mean(jnp.square(score_fake - 1.0))
            score_loss = score_loss / len(disc_fake)

            # Feature Loss
            disc_real= discriminator_state.apply_fn(
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
            # Loss
            loss_g = mel_loss + score_loss +  feat_loss + stft_loss+ loss_kl_f + loss_kl_r * 0.5# + spk_loss * 2

            return loss_g, (fake_audio,audio,mel_loss,stft_loss,loss_kl_f,loss_kl_r,score_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_g,(fake_audio_g,audio_g,mel_loss,stft_loss,loss_kl_f,loss_kl_r,score_loss)), grads_g = grad_fn(generator_state.params)

        # Average across the devices.
        grads_g = jax.lax.pmean(grads_g, axis_name='num_devices')
        loss_g = jax.lax.pmean(loss_g, axis_name='num_devices')
        loss_m = mel_loss.mean()
        loss_s = stft_loss.mean()
        loss_k = loss_kl_f.mean()
        loss_r = loss_kl_r.mean()

        new_generator_state = generator_state.apply_gradients(
            grads=grads_g)#, batch_stats=mutables_g['batch_stats'])
        
        def loss_fn(params):
            disc_fake  = discriminator_state.apply_fn(
                {'params': params},#,'batch_stats': discriminator_state.batch_stats},    
             fake_audio_g)#, mutable=['batch_stats'])
            disc_real  = discriminator_state.apply_fn(
                {'params': params},#,'batch_stats':  mutables['batch_stats']},
                audio_g)#,mutable=['batch_stats'])
            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                loss_d += jnp.mean(jnp.square(score_real - 1.0))
                loss_d += jnp.mean(jnp.square(score_fake))
            loss_d = loss_d / len(disc_fake)
          
            return loss_d#,mutables
        
        # Generate data with the Generator, critique it with the Discriminator.
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

        loss_d, grads_d = grad_fn(discriminator_state.params)

        # Average cross the devices.
        grads_d = jax.lax.pmean(grads_d, axis_name='num_devices')
        loss_d = jax.lax.pmean(loss_d, axis_name='num_devices')

        # Update the discriminator through gradient descent.
        new_discriminator_state = discriminator_state.apply_gradients(
        grads=grads_d)#, batch_stats=mutables_d['batch_stats'])
        return new_generator_state,new_discriminator_state,loss_g,loss_d,loss_m,loss_s,loss_k,loss_r,score_loss
    @partial(jax.pmap, axis_name='num_devices')         
    def do_validate(generator: TrainState,ppg_val:jnp.ndarray,pit_val:jnp.ndarray,spk_val:jnp.ndarray,ppg_l_val:jnp.ndarray,audio:jnp.ndarray):   
        stft = TacotronSTFT(filter_length=hp.data.filter_length,
                hop_length=hp.data.hop_length,
                win_length=hp.data.win_length,
                n_mel_channels=hp.data.mel_channels,
                sampling_rate=hp.data.sampling_rate,
                mel_fmin=hp.data.mel_fmin,
                mel_fmax=hp.data.mel_fmax)      
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp)
        fake_audio = model.apply({'params': generator.params},#,'batch_stats': generator.batch_stats}, 
                                 ppg_val, pit_val, spk_val, ppg_l_val,method=SynthesizerTrn.infer, mutable=False)
        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = stft.mel_spectrogram(audio.squeeze(1))
        mel_loss_val = l1_loss(mel_fake, mel_real)#jnp.mean(optax.l2_loss(mel_fake, mel_real))

        #f idx == 0:
        spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
        spec_real = stft.linear_spectrogram(audio.squeeze(1))
        audio = audio[0][0]
        fake_audio = fake_audio[0][0]
        spec_fake = spec_fake[0]
        spec_real = spec_real[0]
        return mel_loss_val,audio, fake_audio, spec_fake, spec_real
    def validate(generator):
        loader = tqdm.tqdm(valloader, desc='Validation loop')
       
     
        mel_loss = 0.0
        for val_ppg, val_ppg_l, val_pit, val_spk, val_spec, val_spec_l, val_audio, val_audio_l in loader:
            val_ppg=shard(val_ppg)
            val_ppg_l=shard(val_ppg_l)
            val_pit=shard(val_pit)
            val_spk=shard(val_spk)
            val_audio=shard(val_audio)
            mel_loss_val,val_audio,val_fake_audio,spec_fake,spec_real=do_validate(generator,val_ppg,val_pit,val_spk,val_ppg_l,val_audio)
            val_audio,val_fake_audio,spec_fake,spec_real = \
            jax.device_get([val_audio[0],val_fake_audio[0],spec_fake[0],spec_real[0]])
            mel_loss += mel_loss_val.mean()
            writer.log_fig_audio(np.asarray(val_audio), np.asarray(val_fake_audio), \
            np.asarray(spec_fake), np.asarray(spec_real), 0, step)

        mel_loss = mel_loss / len(valloader.dataset)
        mel_loss = np.asarray(mel_loss)
       # (audio,fake_audio,spec_fake,spec_real,idx) = res
       
        writer.log_validation(mel_loss, step)

    key = jax.random.PRNGKey(seed=hp.train.seed)
    key_combine,key_generator, key_discriminator, key = jax.random.split(key, 4)
    key_generator = shard_prng_key(key_generator)
    key_discriminator = shard_prng_key(key_discriminator)
    
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

    discriminator_state = create_discriminator_state(key_discriminator, Discriminator)
    generator_state = create_generator_state(key_generator, SynthesizerTrn)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/sovits5.0/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_g': generator_state, 'model_d': discriminator_state}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        discriminator_state=states['model_d']
        generator_state=states['model_g']

    for epoch in range(init_epoch, hp.train.epochs):

        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        for ppg, ppg_l, pit, spk, spec, spec_l, audio, audio_l in loader:

            ppg = shard(ppg)
            ppg_l = shard(ppg_l)
            pit = shard(pit)
            spk = shard(spk)
            spec = shard(spec)
            spec_l = shard(spec_l)
            audio = shard(audio)
            audio_l = shard(audio_l)
            generator_state,discriminator_state,loss_g,loss_d,loss_m,loss_s,loss_k,loss_r,score_loss=\
            combine_step(generator_state, discriminator_state,ppg=ppg,pit=pit, spk=spk, spec=spec,ppg_l=ppg_l,spec_l=spec_l,audio_e=audio)



            step += 1

            loss_g,loss_d,loss_s,loss_m,loss_k,loss_r,score_loss = \
            jax.device_get([loss_g[0], loss_d[0],loss_s[0],loss_m[0],loss_k[0],loss_r[0],score_loss[0]])
            if rank == 0 and step % hp.log.info_interval == 0:

                writer.log_training(
                    loss_g, loss_d, loss_m, loss_s, loss_k, loss_r, score_loss,step)
                logger.info("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f  | step %d" % (
                    loss_g, loss_m, loss_s, loss_d, loss_k, loss_r, step))
        if rank == 0 and epoch % hp.log.eval_interval == 0:
            validate(generator_state)
        if rank == 0 and epoch % hp.log.save_interval == 0:
            ckpt = {'model_g': generator_state, 'model_d': discriminator_state}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})


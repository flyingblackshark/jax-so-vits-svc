import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import jax
import numpy as np

import argparse
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerTrn
import librosa
from transformers import FlaxAutoModel
import optax
import audax.core
import audax.core.functional
import audax.core.stft
import jax.numpy as jnp
import audax
from librosa.filters import mel as librosa_mel_fn
from jax_fcpe.utils import load_model
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
jax.config.update('jax_platform_name', 'cpu')
WIN_SIZE = 1024
HOP_SIZE = 160
N_FFT = 1024
NUM_MELS = 128
f0_min = 80.
f0_max = 880.
mel_basis = librosa_mel_fn(sr=16000, n_fft=N_FFT, n_mels=NUM_MELS, fmin=0, fmax=8000)
mel_basis = jnp.asarray(mel_basis,dtype=jnp.float32)

def get_f0(wav):
    model,params = load_model()
    wav = jnp.asarray(wav)
    window = jnp.hanning(WIN_SIZE)
    pad_size = (WIN_SIZE-HOP_SIZE)//2
    wav = jnp.pad(wav, ((0,0),(pad_size, pad_size)),mode="reflect")
    spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    spec = spec.transpose(0,2,1)
    mel = jnp.matmul(mel_basis, spec)
    mel = jnp.log(jnp.clip(mel, min=1e-5) * 1)
    mel = mel.transpose(0,2,1)

    def model_predict(mel):
        f0 = model.apply(params,mel,threshold=0.006,method=model.infer)
        uv = (f0 < f0_min).astype(jnp.float32)
        f0 = f0 * (1 - uv)
        return f0
    return model_predict(mel).squeeze(-1)

def speaker2id(hp,key):
    import csv
    reader = csv.reader(open(hp.data.speaker_files, 'r'))
    for row in reader:
        if row[0].lower() == key.lower():
            return int(row[1])
    raise Exception("Speaker Not Found")
def create_generator_state(rng,hp): 
    r"""Create the training state given a model class. """ 
    model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
    segment_size=hp.data.segment_size // hp.data.hop_length,
    hp=hp)
    
    exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.learning_rate, transition_steps=hp.train.total_steps, decay_rate=hp.train.lr_decay)
    optimizer = optax.adamw(learning_rate=exponential_decay_scheduler, b1=hp.train.betas[0],b2=hp.train.betas[1])
    params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
    init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
    example_inputs = {
        "ppg":jnp.zeros((1,400,768)),
        "pit":jnp.zeros((1,400)),
        "spec":jnp.zeros((1,513,400)),
        "ppg_l":jnp.zeros((1),dtype=jnp.int32),
        "spec_l":jnp.zeros((1),dtype=jnp.int32),
        "spk":jnp.ones((1),dtype=jnp.int32),
    }
    

    def init_fn(init_rngs,example_inputs,model, optimizer):
        variables = model.init(init_rngs, **example_inputs,train=False)
        state = TrainState.create( # Create a `TrainState`.
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer)
        return state
    return init_fn(init_rngs, example_inputs,model,optimizer)
def main(args):

    hp = OmegaConf.load(args.config)
    spk = speaker2id(hp,args.spk)
    hubert_model = FlaxAutoModel.from_pretrained("./hubert",from_pt=True, trust_remote_code=True)
    model_g = create_generator_state(jax.random.PRNGKey(0),hp)
    options = ocp.CheckpointManagerOptions(max_to_keep=hp.train.max_to_keep, enable_async_checkpointing=True)
    mngr = ocp.CheckpointManager(hp.log.pth_dir,
    item_names=('model_g', 'model_d'),
    options=options,
    )
    step = mngr.latest_step() 
    states=mngr.restore(step,args=ocp.args.Composite(
    model_g=ocp.args.StandardRestore(model_g)))
    #discriminator_state=states['model_d']
    generator_state=states['model_g']
    
    wav, sr = librosa.load(args.wave, sr=16000)
    pit = get_f0(np.expand_dims(wav,0)).squeeze(0)
    ppg = hubert_model(np.expand_dims(wav,0)).last_hidden_state.squeeze(0)
    ppg = jnp.repeat(ppg,repeats=2,axis=0)
    print("pitch shift: ", args.shift)
    if (args.shift == 0):
        pass
    else:
        pit = np.array(pit)
        source = pit[pit > 0]
        source_ave = source.mean()
        source_min = source.min()
        source_max = source.max()
        print(f"source pitch statics: mean={source_ave:0.1f}, \
                min={source_min:0.1f}, max={source_max:0.1f}")
        shift = args.shift
        shift = 2 ** (shift / 12)
        pit = pit * shift

    len_pit = pit.shape[0]
    len_ppg = ppg.shape[0]
    len_min = min(len_pit, len_ppg)
    pit = pit[:len_min]
    ppg = ppg[:len_min, :]
    
    def parallel_infer(pit_i,ppg_i,spk_i,len_min_i):
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
            segment_size=hp.data.segment_size // hp.data.hop_length,
            hp=hp)
        init_rngs = {'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0),'rnorms':jax.random.PRNGKey(0)}
        out_audio = model.apply( {'params': generator_state.params},ppg_i,pit_i,spk_i,len_min_i,method=SynthesizerTrn.infer,rngs=init_rngs)
        return out_audio
   
    len_min = jnp.asarray([len_min])
    ppg = jnp.asarray(ppg)
    pit = jnp.asarray(pit)
    spk = jnp.asarray(spk)
    ppg = jnp.expand_dims(ppg,0)
    pit = jnp.expand_dims(pit,0)
    spk = jnp.expand_dims(spk,0)

    frags = parallel_infer(pit,ppg,spk,len_min)
    out_audio = jnp.reshape(frags,[frags.shape[0]*frags.shape[1]*frags.shape[2]])

    out_audio = np.asarray(out_audio)
    write("svc_out.wav", 32000, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,  default="./configs/base.yaml",
                        help="yaml file for config.")
    parser.add_argument('--wave', type=str,  default="./test.wav",
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str,  default="AURORA",
                        help="Path of speaker.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)

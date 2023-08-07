import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flax
from flax import linen as nn
from jax.nn.initializers import normal as normal_init
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
import argparse

from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from vits.models import SynthesizerTrn
from pitch import load_csv_pitch
from flax.training import orbax_utils
import orbax
from flax.training.common_utils import shard



def main(args):

    if (args.ppg == None):
        args.ppg = "svc_tmp.ppg.npy"
        print(
            f"Auto run : python3 whisper/inference.py -w {args.wave} -p {args.ppg}")
        os.system(f"python3 whisper/inference.py -w {args.wave} -p {args.ppg}")
    if (args.vec == None):
        args.vec = "svc_tmp.vec.npy"
        print(
            f"Auto run : python hubert/inference.py -w {args.wave} -v {args.vec}")
        os.system(f"python3 hubert/inference.py -w {args.wave} -v {args.vec}")
    if (args.pit == None):
        args.pit = "svc_tmp.pit.csv"
        print(
            f"Auto run : python3 pitch/inference.py -w {args.wave} -p {args.pit}")
        os.system(f"python3 pitch/inference.py -w {args.wave} -p {args.pit}")

    
    hp = OmegaConf.load(args.config)
    def create_generator_state(): 
        r"""Create the training state given a model class. """ 
        rng = jax.random.PRNGKey(1234)
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
        segment_size=hp.data.segment_size // hp.data.hop_length,
        hp=hp)
        

        tx = optax.lion(learning_rate=0.01, b1=hp.train.betas[0],b2=hp.train.betas[1])
          
        #(fake_ppg,fake_ppg_l,fake_vec,fake_pit,fake_spk,fake_spec,fake_spec_l,fake_audio,wav_l) = next(iter(trainloader))
        fake_ppg = jnp.ones((1,400,1280))
        fake_vec = jnp.ones((1,400,256))
        fake_spec = jnp.ones((1,513,400))
        fake_ppg_l = jnp.ones((1))
        fake_spec_l = jnp.ones((1))
        fake_pit = jnp.ones((1,400))
        fake_spk = jnp.ones((1,256))
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        
        variables = model.init(init_rngs, ppg=fake_ppg, pit=fake_pit,vec=fake_vec, spec=fake_spec, spk=fake_spk, ppg_l=fake_ppg_l, spec_l=fake_spec_l,train=False)

        state = TrainState.create(apply_fn=SynthesizerTrn.apply, tx=tx,params=variables['params'])
        
        return state
    generator_state = create_generator_state()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        'chkpt/sovits5.0/', orbax_checkpointer, options)
    if checkpoint_manager.latest_step() is not None:
        target = {'model_g': generator_state, 'model_d': None}
        step = checkpoint_manager.latest_step()  # step = 4
        states=checkpoint_manager.restore(step,items=target)
        #discriminator_state=states['model_d']
        generator_state=states['model_g']

    spk = np.load(args.spk)
    ppg = np.load(args.ppg)
    ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
    vec = np.load(args.vec)
    vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
    pit = load_csv_pitch(args.pit)
    pit = np.array(pit)
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
    vec = vec[:len_min, :]
    ppg = ppg[:len_min, :]
    pad_to_device_num = (8 - len_min%8)
    len_min = len_min + pad_to_device_num
    pit = jnp.pad(pit,[(0,pad_to_device_num)])
    vec = jnp.pad(vec,[(0,pad_to_device_num),(0,0)])
    ppg = jnp.pad(ppg,[(0,pad_to_device_num),(0,0)])
    pit = jnp.stack(jnp.split(pit,8))
    vec = jnp.stack(jnp.split(vec,8))
    ppg = jnp.stack(jnp.split(ppg,8))
    spk = jnp.asarray(spk)
    spk = jnp.expand_dims(spk,0)
    spk = jnp.broadcast_to(spk,[8,256])
    len_min = len_min/8
    len_min = jnp.asarray([len_min])
    len_min = jnp.broadcast_to(len_min,[8])
    @jax.pmap
    def parallel_infer(pit_i,ppg_i,spk_i,vec_i,len_min_i):
        model = SynthesizerTrn(spec_channels=hp.data.filter_length // 2 + 1,
            segment_size=hp.data.segment_size // hp.data.hop_length,
            hp=hp,train=False)
        rng = jax.random.PRNGKey(1234)
        params_key,r_key,dropout_key,rng = jax.random.split(rng,4)
        init_rngs = {'params': params_key, 'dropout': dropout_key,'rnorms':r_key}
        out_audio = model.apply( {'params': generator_state.params},ppg_i,pit_i,vec_i,spk_i,len_min_i,method=SynthesizerTrn.infer,rngs=init_rngs)
        return out_audio
        #out_audio = np.asarray(out_audio)
    print(ppg.shape)
    print(pit.shape)
    print(spk.shape)
    print(vec.shape)
    print(len_min.shape)
    pit=shard(pit)
    ppg=shard(ppg)
    spk=shard(spk)
    vec=shard(vec)
    len_min=shard(len_min)
    frags = parallel_infer(pit,ppg,spk,vec,len_min)
    out_audio = jnp.reshape(frags,[frags.shape[0]*frags.shape[1]*frags.shape[2]*frags.shape[3]])
    #out_audio = jnp.concatenate(frags,0)
    # spk = spk.unsqueeze(0).to(device)
    # source = pit.unsqueeze(0).to(device)
    # source = model.pitch2source(source)
    # pitwav = model.source2wav(source)
    # write("svc_out_pit.wav", hp.data.sampling_rate, pitwav)

    # hop_size = hp.data.hop_length
    # all_frame = len_min
    # hop_frame = 10
    # out_chunk = 2500  # 25 S
    # out_index = 0
    # out_audio = []
    # has_audio = False

    # while (out_index + out_chunk < all_frame):
    #     has_audio = True
    #     if (out_index == 0):  # start frame
    #         cut_s = 0
    #         cut_s_out = 0
    #     else:
    #         cut_s = out_index - hop_frame
    #         cut_s_out = hop_frame * hop_size

    #     if (out_index + out_chunk + hop_frame > all_frame):  # end frame
    #         cut_e = out_index + out_chunk
    #         cut_e_out = 0
    #     else:
    #         cut_e = out_index + out_chunk + hop_frame
    #         cut_e_out = -1 * hop_frame * hop_size

    #     sub_ppg = ppg[cut_s:cut_e, :].unsqueeze(0).to(device)
    #     sub_pit = pit[cut_s:cut_e].unsqueeze(0).to(device)
    #     sub_len = torch.LongTensor([cut_e - cut_s]).to(device)
    #     sub_har = source[:, :, cut_s *
    #                         hop_size:cut_e * hop_size].to(device)
    #     sub_out = model.inference(sub_ppg, sub_pit, spk, sub_len, sub_har)
    #     sub_out = sub_out[0, 0].data.cpu().detach().numpy()

    #     sub_out = sub_out[cut_s_out:cut_e_out]
    #     out_audio.extend(sub_out)
    #     out_index = out_index + out_chunk

    # if (out_index < all_frame):
    #     if (has_audio):
    #         cut_s = out_index - hop_frame
    #         cut_s_out = hop_frame * hop_size
    #     else:
    #         cut_s = 0
    #         cut_s_out = 0
    #     sub_ppg = ppg[cut_s:, :].unsqueeze(0).to(device)
    #     sub_pit = pit[cut_s:].unsqueeze(0).to(device)
    #     sub_len = torch.LongTensor([all_frame - cut_s]).to(device)
    #     sub_har = source[:, :, cut_s * hop_size:].to(device)
    #     sub_out = model.inference(sub_ppg, sub_pit, spk, sub_len, sub_har)
    #     sub_out = sub_out[0, 0].data.cpu().detach().numpy()

    #     sub_out = sub_out[cut_s_out:]
    #     out_audio.extend(sub_out)
    out_audio = np.asarray(out_audio)
    write("svc_out.wav", 32000, out_audio)
    #write("svc_out.wav", hp.data.sampling_rate, out_audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help="yaml file for config.")
    # parser.add_argument('--model', type=str, required=True,
    #                     help="path of model for evaluation")
    parser.add_argument('--wave', type=str, required=True,
                        help="Path of raw audio.")
    parser.add_argument('--spk', type=str, required=True,
                        help="Path of speaker.")
    parser.add_argument('--ppg', type=str,
                        help="Path of content vector.")
    parser.add_argument('--vec', type=str,
                        help="Path of hubert vector.")
    parser.add_argument('--pit', type=str,
                        help="Path of pitch csv file.")
    parser.add_argument('--shift', type=int, default=0,
                        help="Pitch shift key.")
    args = parser.parse_args()

    main(args)

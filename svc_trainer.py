import argparse
from omegaconf import OmegaConf
from vits_extend.train import train
import jax

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils



if __name__ == '__main__':
    #jax.distributed.initialize()
    device_mesh = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))   
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="configs/base.yaml",
                        help="yaml file for configuration")
    # parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
    #                     help="path of checkpoint pt file to resume training")
    # parser.add_argument('-n', '--name', type=str, default="sovits",
    #                     help="name of the model for logging, saving checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length


    train(args, hp , mesh)
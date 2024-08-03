
import datasets
import grain.python
import utils
import grain
from jax.sharding import Mesh
import jax
import multihost_dataloading
import os
from transformers import FlaxAutoModel
import glob
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
jax.config.update('jax_platform_name', 'cpu')

def main():
    data_files = glob.glob("dataset/aurora/*.arrayrecord")
    dataset = grain.python.ArrayRecordDataSource(data_files)
    # train_ds = datasets.load_dataset(
    # "../aurora",
    # #"fbs0/aurora",
    # #data_dir=config.hf_data_dir,
    # #data_files=config.hf_train_files,
    # split="train",
    # streaming=True,
    # token="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",
    # )
    global_mesh = Mesh(jax.devices(),('x'))
    #dataset = utils.HFDataSource(train_ds, jax.process_index(), 2, 1)
    index_sampler = grain.python.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.python.ShardOptions(
          shard_index=jax.process_index(), shard_count=1, drop_remainder=True
      ),
      shuffle=False,
      seed=0,
    )
    # dummy_index_sampler = grain.python.IndexSampler(
    #     num_records=len(dataset),
    #     num_epochs=1,
    #     shard_options=grain.python.ShardOptions(
    #         shard_index=jax.process_index(), shard_count=2, drop_remainder=False
    #     ),
    #     shuffle=False,
    #     seed=0,
    # )
    operations = []
    #hubert_model = FlaxAutoModel.from_pretrained("./hubert",from_pt=True, trust_remote_code=True)
    #operations.append(utils.PreprocessAudioFiles(hubert_model))
    #operations.append(utils.PadToMaxLength(15*44100,749,1501,1498))
    
    operations.append(utils.ParseFeatures())
    operations.append(utils.PadToMaxLength(15*44100,750,1500,1500))
    operations.append(grain.python.Batch(batch_size=4 // jax.process_count(), drop_remainder=True))
    dataloader = grain.python.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=index_sampler,
        worker_count=1,  # only supports one worker for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.python.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)
    #if jax.process_index() == 0:
    data = next(multihost_gen)
    print(data)
if __name__ == "__main__":
    main()
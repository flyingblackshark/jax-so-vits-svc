import grain.python
from input_pipeline import utils
import grain
from jax.sharding import Mesh
import jax
from input_pipeline import multihost_dataloading
from transformers import FlaxAutoModel
import glob
def get_dataset(hp):
    data_files = glob.glob("dataset/aurora/*.arrayrecord")
    dataset = grain.python.ArrayRecordDataSource(data_files)
    index_sampler = grain.python.IndexSampler(
      num_records=len(dataset),
      num_epochs=hp.data_loader.num_epochs,
      shard_options=grain.python.ShardOptions(
          shard_index=jax.process_index(), shard_count=hp.data_loader.host_number, drop_remainder=True
      ),
      shuffle=True,
      seed=0,
    )
    global_mesh = Mesh(jax.devices(),('x'))
    operations = []
    operations.append(utils.ParseFeatures(hp))
    operations.append(utils.PadToMaxLength(15*44100,1500,1500,1500))
    operations.append(grain.python.Batch(batch_size=hp.data_loader.global_batch_size // jax.process_count(), drop_remainder=True))
    dataloader = grain.python.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=index_sampler,
        worker_count=hp.data_loader.worker_count
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)
    return multihost_gen
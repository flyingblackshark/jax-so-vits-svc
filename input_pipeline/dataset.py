import datasets
import grain.python
from input_pipeline import utils
import grain
from jax.sharding import Mesh
import jax
from input_pipeline import multihost_dataloading
from transformers import FlaxAutoModel
def get_dataset():
    train_ds = datasets.load_dataset(
    "../aurora",
    #"fbs0/aurora",
    #data_dir=config.hf_data_dir,
    #data_files=config.hf_train_files,
    split="train",
    streaming=True,
    token="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",
    )
    global_mesh = Mesh(jax.devices(),('x'))
    dataset = utils.HFDataSource(train_ds, jax.process_index(), 2, 1)
    dummy_index_sampler = grain.python.IndexSampler(
        num_records=len(dataset),
        num_epochs=1,
        shard_options=grain.python.ShardOptions(
            shard_index=jax.process_index(), shard_count=2, drop_remainder=False
        ),
        shuffle=False,
        seed=0,
    )
    operations = []
    hubert_model = FlaxAutoModel.from_pretrained("./hubert",from_pt=True, trust_remote_code=True)
    operations.append(utils.PreprocessAudioFiles(hubert_model))
    #operations.append(utils.PadToMaxLength(15*44100,749,1501,1498))
    operations.append(utils.PadToMaxLength(15*44100,750,1500,1500))
    operations.append(grain.python.Batch(batch_size=4 // jax.process_count(), drop_remainder=True))
    dataloader = grain.python.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=1,  # only supports one worker for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.python.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)
    return multihost_gen
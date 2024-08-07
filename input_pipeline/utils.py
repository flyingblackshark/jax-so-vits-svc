#import datasets
import grain
import warnings
#from datasets.distributed import split_dataset_by_node
from threading import current_thread
import numpy as np
import grain.python
from input_pipeline import max_logging
#import max_logging
#import librosa
#import torchcrepe
#import scipy
#import torch
import tensorflow as tf
import random

# class HFDataSource(grain.python.RandomAccessDataSource):
#   """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

#   def __init__(self, dataset: datasets.IterableDataset, dataloading_host_index: int, dataloading_host_count: int, num_threads: int):
#     self.dataset = dataset
#     self.num_threads = num_threads
#     self.dataloading_host_count = dataloading_host_count
#     self.dataloading_host_index = dataloading_host_index
#     self.n_shards = dataset.n_shards
#     self._check_shard_count()
#     self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
#     self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
#     self.data_iters = []
#     self.out_of_data =False

#   def _check_shard_count(self):
#     if self.n_shards < (self.dataloading_host_count * self.num_threads):
#       warnings.warn(f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
#                       "smaller than number of host loading data. This is known to lead to inefficient dataloading. " 
#                       "Please reshard the data, or use a subset of hosts for dataloading by setting expansion_factor_real_data."
#                       "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
#                       )
#       self.n_shards = self.dataloading_host_count * self.num_threads
#     elif self.n_shards % (self.dataloading_host_count * self.num_threads) > 0:
#       usable_shards = (
#           self.n_shards
#           // (self.dataloading_host_count * self.num_threads)
#           * (self.dataloading_host_count * self.num_threads)
#       )
#       warnings.warn(f"Dataset contains {self.n_shards} shards, but only {usable_shards} shards will be used."
#                     "Make (dataset shards) % (number of host loading data) == 0 to use all shards of data"
#                     "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
#                     )
#   def _check_shard_count(self):
#     if self.n_shards < (self.dataloading_host_count * self.num_threads):
#       warnings.warn(f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
#                       "smaller than number of host loading data. This is known to lead to inefficient dataloading. " 
#                       "Please reshard the data, or use a subset of hosts for dataloading by setting expansion_factor_real_data."
#                       "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
#                       )
#       self.n_shards = self.dataloading_host_count * self.num_threads
#     elif self.n_shards % (self.dataloading_host_count * self.num_threads) > 0:
#       usable_shards = (
#           self.n_shards
#           // (self.dataloading_host_count * self.num_threads)
#           * (self.dataloading_host_count * self.num_threads)
#       )
#       warnings.warn(f"Dataset contains {self.n_shards} shards, but only {usable_shards} shards will be used."
#                     "Make (dataset shards) % (number of host loading data) == 0 to use all shards of data"
#                     "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
#                     )

#   def _update_shard(self, idx):
#     if self.dataset_shards[idx] < self.n_shards:
#       max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
#       self.dataset_shards[idx] += self.dataloading_host_count * self.num_threads
#       max_logging.log(f"New shard is {self.dataset_shards[idx]}")
#       self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
#       self.data_iters[idx] = iter(self.datasets[idx])
#     else:
#       max_logging.log(f"Run out of shards on host {self.dataloading_host_index}, shard {self.dataset_shards[idx]} is not available")
#       self.out_of_data = True


#   def __len__(self):
#     """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
#     a fake length bigger than the dataset is returned"""
#     return 10_000_000_000

#   def __getitem__(self, index):
#     """Since HuggingFace IterableDataset does not support random access by index.
#     The next item in the iterator is returned."""
#     if not self.data_iters:
#       self.data_iters = [iter(x) for x in self.datasets]
#     idx = int(current_thread().name.split("_")[1])

#     while True:
#       try:
#         if self.out_of_data:
#           return None
#         data = next(self.data_iters[idx])
#         return data
#       except StopIteration:
#         self._update_shard(idx)
class SliceToLength(grain.python.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self,hp):
    self.use = int(hp.data.segment_size / hp.data.hop_length * 16)  # 4 S
    self.hop_length = hp.data.hop_length
  def map(self, data):
    max_frame_start = data["hubert_feature"].shape[0] - self.use - 1
    frame_start = random.randint(0, int(max_frame_start))
    frame_end = frame_start + self.use
    
    data["hubert_feature"] = data["hubert_feature"][frame_start:frame_end, :]
    data["f0_feature"] = data["f0_feature"][frame_start:frame_end]
    data["spec_feature"] = data["spec_feature"][:, frame_start:frame_end]

    wav_start = frame_start * self.hop_length
    wav_end = frame_end * self.hop_length
    data["audio"] = data["audio"][wav_start:wav_end]
    return data
class PadToMaxLength(grain.python.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self, 
               audio_max_length,
               hubert_max_length,
               f0_max_length,
               spec_max_length):
    self.audio_max_length = audio_max_length
    self.hubert_max_length = hubert_max_length
    self.f0_max_length = f0_max_length
    self.spec_max_length = spec_max_length

  def map(self, data):
    """map to each element"""

    def pad_audio(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)]
      return np.pad(x, pad_amount)
    def pad_hubert(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      return np.pad(x, ((0,pad_amount),(0,0)))
    def pad_f0(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      return np.pad(x, ((0,pad_amount)))
    def pad_spec(x, max_length):
      pad_amount = max(max_length - x.shape[1], 0)
      return np.pad(x, ((0,0),(0,pad_amount)))
    
    data["audio"] = pad_audio(data["audio"], self.audio_max_length)
    data["hubert_feature"] = pad_hubert(data["hubert_feature"], self.hubert_max_length)
    data["f0_feature"] = pad_f0(data["f0_feature"], self.f0_max_length)
    data["spec_feature"] = pad_spec(data["spec_feature"], self.spec_max_length)
    return data
  

class ParseFeatures(grain.python.MapTransform):
  # """Parse serialized example"""
  def __init__(self, hp):
    self.hp = hp
  def speaker2id(self,key):
    import csv
    reader = csv.reader(open(self.hp.data.speaker_files, 'r'))
    for row in reader:
      if row[0].lower() == key:
      #if (tf.strings.unicode_decode(row[0].lower(),"UTF-8").numpy() == key.numpy()).all():
        return int(row[1])
    raise Exception("Speaker Not Found")
  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(example, {
        "audio": tf.io.FixedLenFeature([], dtype=tf.string),
        "spec_feature": tf.io.FixedLenFeature([], dtype=tf.string),
        "f0_feature": tf.io.FixedLenFeature([], dtype=tf.string),
        "hubert_feature": tf.io.FixedLenFeature([], dtype=tf.string),
        "speaker": tf.io.FixedLenFeature([], dtype=tf.string)
        })
      return parsed
    example = _parse(features)
    audio = tf.io.parse_tensor(example["audio"],tf.float32)
    hubert_feature = tf.io.parse_tensor(example["hubert_feature"],tf.float32)
    f0_feature = tf.io.parse_tensor(example["f0_feature"],tf.float32)
    spec_feature = tf.io.parse_tensor(example["spec_feature"],tf.float32)

    hubert_feature = tf.repeat(hubert_feature,repeats=2,axis=0) #replicate
    #speaker = tf.strings.unicode_decode(example["speaker"],'UTF-8')
    #return _parse(features)
    return {
        "audio": audio,
        "audio_length":audio.shape[0],
        "hubert_feature": hubert_feature,
        "hubert_length":hubert_feature.shape[0],
        "f0_feature": f0_feature,
        "f0_length": f0_feature.shape[0],
        "spec_feature":spec_feature,
        "spec_length": spec_feature.shape[1],
        "speaker_id":self.speaker2id(example["speaker"])
    }
  
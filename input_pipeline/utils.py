import datasets
import grain
import warnings
from datasets.distributed import split_dataset_by_node
from threading import current_thread
import numpy as np
import grain.python
from input_pipeline import max_logging
import librosa
import torchcrepe
import scipy
import torch
class HFDataSource(grain.python.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(self, dataset: datasets.IterableDataset, dataloading_host_index: int, dataloading_host_count: int, num_threads: int):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []
    self.out_of_data =False

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
                      "smaller than number of host loading data. This is known to lead to inefficient dataloading. " 
                      "Please reshard the data, or use a subset of hosts for dataloading by setting expansion_factor_real_data."
                      "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
                      )
      self.n_shards = self.dataloading_host_count * self.num_threads
    elif self.n_shards % (self.dataloading_host_count * self.num_threads) > 0:
      usable_shards = (
          self.n_shards
          // (self.dataloading_host_count * self.num_threads)
          * (self.dataloading_host_count * self.num_threads)
      )
      warnings.warn(f"Dataset contains {self.n_shards} shards, but only {usable_shards} shards will be used."
                    "Make (dataset shards) % (number of host loading data) == 0 to use all shards of data"
                    "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
                    )
  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
                      "smaller than number of host loading data. This is known to lead to inefficient dataloading. " 
                      "Please reshard the data, or use a subset of hosts for dataloading by setting expansion_factor_real_data."
                      "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
                      )
      self.n_shards = self.dataloading_host_count * self.num_threads
    elif self.n_shards % (self.dataloading_host_count * self.num_threads) > 0:
      usable_shards = (
          self.n_shards
          // (self.dataloading_host_count * self.num_threads)
          * (self.dataloading_host_count * self.num_threads)
      )
      warnings.warn(f"Dataset contains {self.n_shards} shards, but only {usable_shards} shards will be used."
                    "Make (dataset shards) % (number of host loading data) == 0 to use all shards of data"
                    "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#limitations--recommendations"
                    )

  def _update_shard(self, idx):
    if self.dataset_shards[idx] < self.n_shards:
      max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
      self.dataset_shards[idx] += self.dataloading_host_count * self.num_threads
      max_logging.log(f"New shard is {self.dataset_shards[idx]}")
      self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      max_logging.log(f"Run out of shards on host {self.dataloading_host_index}, shard {self.dataset_shards[idx]} is not available")
      self.out_of_data = True


  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        if self.out_of_data:
          return None
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)

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
  
class PreprocessAudioFiles(grain.python.MapTransform):
  """Normalize feature keys for HuggingFace input"""
  def __init__(self, hubert_model):
    self.hubert_model = hubert_model
  def predict_f0(self,audio):
    audio = torch.tensor(np.copy(audio),dtype=torch.float32)[None]
    hop_length = 160
    fmin = 50
    fmax = 1000
    model = "full"
    batch_size = 1
    sr = 16000
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device="cpu",
        return_periodicity=True,
    )
    # CREPE was not trained on silent audio. some error on silent need filter.pitPath
    periodicity = torchcrepe.filter.median(periodicity, 7)
    pitch = torchcrepe.filter.mean(pitch, 5)
    pitch[periodicity < 0.5] = 0
    pitch = pitch.squeeze(0)
    return pitch.numpy()
  def spectrogram(self,y, n_fft, hop_size, win_size):
    spec = scipy.signal.stft(y, nfft=n_fft, noverlap=win_size-hop_size, nperseg=win_size,return_onesided=True,padded=True,boundary=None)
    spec = np.abs(spec[2])+(1e-9)
    return spec.squeeze(0)
  def compute_spec(self,audio):
    audio_norm = audio / 32768.0
    audio_norm = np.expand_dims(audio_norm,axis=0)
    n_fft = 1024
    hop_size = 320
    win_size = 1024
    spec = self.spectrogram(audio_norm, n_fft, hop_size, win_size)
    return spec
  def map(self, features):
    audio_arr_44k = np.asarray(features["audio"]["array"])
    audio_arr_32k = librosa.resample(audio_arr_44k, orig_sr=features["audio"]["sampling_rate"], target_sr=32000)
    audio_arr_16k = librosa.resample(audio_arr_44k, orig_sr=features["audio"]["sampling_rate"], target_sr=16000)
    hubert_feature = self.hubert_model(np.expand_dims(audio_arr_16k,0)).last_hidden_state.squeeze(0)
    
    return {
        "audio": audio_arr_32k,
        "hubert_feature":hubert_feature,
        "f0_feature":self.predict_f0(audio_arr_16k),
        "spec_feature":self.compute_spec(audio_arr_32k)
    }


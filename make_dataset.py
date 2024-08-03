import tensorflow as tf
import numpy as np
import argparse
import os
import librosa
import torch
import torchcrepe
from array_record.python.array_record_module import ArrayRecordWriter
import jax
import jax.numpy as jnp
from transformers import FlaxAutoModel
def write_example_to_arrayrecord(example, file_path):
  writer = ArrayRecordWriter(file_path, 'group_size:1')
  writer.write(example.SerializeToString())
  writer.close()  
def predict_f0(audio):
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
def spectrogram(y, n_fft, hop_size, win_size):
    spec = jax.scipy.signal.stft(y, nfft=n_fft, noverlap=win_size-hop_size, nperseg=win_size,return_onesided=True,padded=True,boundary=None)
    spec = np.abs(spec[2])+(1e-9)
    return spec.squeeze(0)
def compute_spec(audio):
    audio_norm = audio / 32768.0
    audio_norm = jnp.expand_dims(audio_norm,axis=0)
    n_fft = 1024
    hop_size = 320
    win_size = 1024
    spec = spectrogram(audio_norm, n_fft, hop_size, win_size)
    return spec
def process_file(wavPath,outPath,spks,hubert_model):
    for file in os.listdir(f"./{wavPath}/{spks}"):
        if file.endswith(".wav"):
            file = file[:-4]
            wav, sr = librosa.load(f"{wavPath}/{spks}/{file}.wav", sr=44100)
            audio_arr_32k = librosa.resample(wav, orig_sr=sr, target_sr=32000)
            audio_arr_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            hubert_feature = hubert_model(np.expand_dims(audio_arr_16k,0)).last_hidden_state.squeeze(0)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'audio': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(wav).numpy()])),
                        'spec_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(compute_spec(audio_arr_32k)).numpy()])),
                        'f0_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(predict_f0(audio_arr_16k)).numpy()])),
                        'hubert_feature': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(hubert_feature).numpy()])),
                        'speaker':tf.train.Feature(bytes_list=tf.train.BytesList(value=[spks.encode('utf-8')]))
                    }
                )
            )
            write_example_to_arrayrecord(example=example,file_path=f"{outPath}/{spks}/{file}.arrayrecord")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    #parser.add_argument("-s", "--spk", help="spk", dest="spk", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    #parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)

    args = parser.parse_args()
    hubert_model = FlaxAutoModel.from_pretrained("./hubert",from_pt=True, trust_remote_code=True)
    # print(args.wav)
    # print(args.out)
    # print(args.sr)

    #os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out
    #spkPath = args.spk

    # reader = csv.reader(open(spkPath, 'r'))
    # spk_dict = {}
    # for row in reader:
    #     k, v = row
    #     spk_dict[k.lower()] = v
    #assert args.sr == 16000 or args.sr == 32000
    
    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{outPath}/{spks}", exist_ok=True)
            process_file(wavPath,outPath,spks,hubert_model)
            # if args.thread_count == 0:
            #     process_num = os.cpu_count() // 2 + 1
            # else:
            #     process_num = args.thread_count
            # process_files_with_thread_pool(wavPath, spks, outPath, args.sr, process_num)
    # speaker = "AURORA"
    # for i in range(10):
    #     write_example_to_arrayrecord(example=example,file_path=f"{speaker}_{i}.arrayrecord")
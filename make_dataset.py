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
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import FlaxAutoModel
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
jax.config.update('jax_platform_name', 'cpu')
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
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
    batch_size = 512
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
    f,t,spec = jax.scipy.signal.stft(y, nfft=n_fft, noverlap=win_size-hop_size, nperseg=win_size,return_onesided=True,padded=True,boundary=None)
    spec =  (spec.imag**2 + spec.real**2)+(1e-9)
    return spec.squeeze(0)
def compute_spec(audio):
    audio_norm = audio / 32768.0
    audio_norm = jnp.expand_dims(audio_norm,axis=0)
    n_fft = 1024
    hop_size = 320
    win_size = 1024
    spec = spectrogram(audio_norm, n_fft, hop_size, win_size)
    return spec
# def process_files_with_thread_pool(wavPath, spks, outPath, thread_num,hubert_model):
#     files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

#     with ThreadPoolExecutor(max_workers=thread_num) as executor:
#         futures = {executor.submit(process_file, file, wavPath, spks, outPath, hubert_model): file for file in files}

#         for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {sr} {spks}'):
#             future.result()
def process_file(wavPath,spks,outPath,hubert_model):
    for file in os.listdir(f"./{wavPath}/{spks}"):
        if file.endswith(".wav"):
            file = file[:-4]
            wav, sr = librosa.load(f"{wavPath}/{spks}/{file}.wav", sr=44100)
            audio_arr_32k = librosa.resample(wav, orig_sr=sr, target_sr=32000)
            audio_arr_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            test_shape = jax.eval_shape(hubert_model,jax.ShapeDtypeStruct(np.expand_dims(audio_arr_16k,0).shape, jnp.float32))
            now_l = test_shape.last_hidden_state.shape[1]
            audio_arr_16k_padded = np.pad(audio_arr_16k,(0,30*16000-audio_arr_16k.shape[0]))
            hubert_feature = jax.jit(hubert_model)(np.expand_dims(audio_arr_16k_padded,0)).last_hidden_state.squeeze(0)
            hubert_feature = hubert_feature[:now_l]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'audio': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(audio_arr_32k).numpy()])),
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
            print("write "+f"{outPath}/{spks}/{file}.arrayrecord")
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
            process_file(wavPath,spks,outPath,hubert_model)
            # if args.thread_count == 0:
            #     process_num = os.cpu_count() // 2 + 1
            # else:
            #     process_num = args.thread_count
            # process_files_with_thread_pool(wavPath, spks, outPath, args.sr, process_num)
    # speaker = "AURORA"
    # for i in range(10):
    #     write_example_to_arrayrecord(example=example,file_path=f"{speaker}_{i}.arrayrecord")
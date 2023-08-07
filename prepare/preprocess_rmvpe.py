import os
import numpy as np
import librosa
import argparse
from rmvpe import RMVPE
import torch
model_rmvpe = RMVPE(
        "rmvpe.pt", is_half=False, device=torch.device("cpu")
    )
def compute_f0(path, save):
    x, sr = librosa.load(path, sr=16000)
    assert sr == 16000
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    np.save(save, f0, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)
    os.makedirs(args.pit)
    wavPath = args.wav
    pitPath = args.pit

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    print(file)
                    file = file[:-4]
                    compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit")
        else:
            file = spks
            if file.endswith(".wav"):
                print(file)
                file = file[:-4]
                compute_f0(f"{wavPath}/{file}.wav", f"{pitPath}/{file}.pit")

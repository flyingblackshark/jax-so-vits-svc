import os
import random


def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


if __name__ == "__main__":
    os.makedirs("./files/", exist_ok=True)

    rootPath = "./data_svc/waves-32k/"
    all_items = []
    i=0
    for spks in os.listdir(f"./{rootPath}"):
        i+=1
        if not os.path.isdir(f"./{rootPath}/{spks}"):
            continue
        print(f"./{rootPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith(".wav"):
                file = file[:-4]
                path_wave = f"./data_svc/waves-32k/{spks}/{file}.wav"
                path_spec = f"./data_svc/specs/{spks}/{file}.pt.npy"
                path_pitch = f"./data_svc/pitch/{spks}/{file}.pit.npy"
                path_hubert = f"./data_svc/hubert/{spks}/{file}.vec.npy"
                has_error = 0
                if not os.path.isfile(path_wave):
                    print_error(path_wave)
                    has_error = 1
                if not os.path.isfile(path_spec):
                    print_error(path_spec)
                    has_error = 1
                if not os.path.isfile(path_pitch):
                    print_error(path_pitch)
                    has_error = 1
                if not os.path.isfile(path_hubert):
                    print_error(path_hubert)
                    has_error = 1
                if has_error == 0:
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_hubert}|{i}")

    random.shuffle(all_items)
    valids = all_items[:8]
    valids.sort()
    trains = all_items[8:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()

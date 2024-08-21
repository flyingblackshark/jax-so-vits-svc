export PYTHONPATH=$PWD
python prepare/resample.py -w ./dataset_raw -o ./dataset/waves-16k -s 16000 -t 12
python prepare/resample.py -w ./dataset_raw -o ./dataset/waves-32k -s 32000 -t 12
python prepare/gen_f0.py -w dataset/waves-16k/ -o dataset/pitch
python prepare/gen_hubert.py -w dataset/waves-16k/ -o dataset/hubert
python prepare/gen_spec.py -w dataset/waves-32k/ -o dataset/spec
python make_dataset_from_dataset.py -l ./dataset_raw -d ./dataset -o ./processed
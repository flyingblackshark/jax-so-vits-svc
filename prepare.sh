export PYTHONPATH=$PWD
python3 prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000 -t 240
python3 prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-32k -s 32000 -t 240
python3 prepare/preprocess_crepe.py -w data_svc/waves-16k/ -p data_svc/pitch
python3 prepare/preprocess_hubert.py -w data_svc/waves-16k/ -v data_svc/hubert
python3 prepare/preprocess_spec_jax.py -w data_svc/waves-32k/ -s data_svc/specs
python3 prepare/preprocess_train_jax.py

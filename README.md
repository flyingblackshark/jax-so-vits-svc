# SO-VITS-SVC 5.0 IN JAX
The following tutorials are for Google TPU v2-8/v3-8

## First Step (Prepare)
	pip install -r requirements.txt
	sudo apt install -y libsndfile1 ffmpeg
## Second Step
Dwonload pretrained models from [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)
Generate or copy your data_svc folder from [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)
## Third Step (Train)
	python3 svc_trainer.py
## Fourth Step (Inference)
	python3 svc_inference.py --config configs/base.yaml --spk xxx.spk.npy --wave test.wav
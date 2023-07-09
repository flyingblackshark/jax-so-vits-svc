# SO-VITS-SVC 5.0 IN JAX
## 第一步
利用你的CPU或者GPU生成data_svc训练集，或者直接迁移[so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)的训练集
## 配置环境
	pip install -r requirements.txt
	sudo apt install -y libsndfile1 ffmpeg
## 开始训练
	python3 svc_trainer.py
## 推理
	python3 svc_inference.py --config configs/base.yaml --spk xxx.spk.npy --wave test.wav
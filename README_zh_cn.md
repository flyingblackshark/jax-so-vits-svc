[English](https://github.com/flyingblackshark/jax-so-vits-svc-5.0/blob/Main-5/README.md) | 中文教程
# SO-VITS-SVC 5.0 IN JAX
以下教程针对谷歌TPU v2-8/v3-8

## 配置环境
	pip install -r requirements.txt
	pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
	sudo apt update && sudo apt install -y libsndfile1 ffmpeg
## 制作数据集
从 [so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)下载HuBERT\Whisper\Speaker预训练模型
利用你的CPU或者GPU生成data_svc训练集，或者直接从[so-vits-svc-5.0](https://github.com/PlayVoice/so-vits-svc-5.0)的训练集迁移
将原项目中的HuBERT预训练模型替换成 [最新模型](https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt)
## 开始训练
	python3 prepare/preprocess_train.py
	python3 svc_trainer.py
## 推理
	python3 svc_inference.py --config configs/base.yaml --spk xxx.spk.npy --wave test.wav

### 讨论QQ群 771728973
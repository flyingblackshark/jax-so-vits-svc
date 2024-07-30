pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
wget https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/pytorch_model.bin?download=true -o ./hubert/pytorch_model.bin
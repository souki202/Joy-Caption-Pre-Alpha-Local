---
title: Joy Caption Pre Alpha
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

## Original Repository

[fancyfeast/joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)

## Requirements

10~12GB VRAM in 8bit.
20GB VRAM in 16bit.

## Usage

Intended for use with WSL.
You must obtain Llama3.1-8B access on HuggingFace in advance.

1. Install the requirements:

```sh
python -m venv venv
source venv/bin/activate

# Install the requirements. The CUDA version should be appropriate for your environment.
pip install torch==2.4.0 torchvision==2.4.0 torchaudio==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

2. Run the app:

```sh
# Run on 16bit
python app.py --token YOUR_HF_TOKEN

# Run on 8bit
python app.py --token YOUR_HF_TOKEN --use-8bit

# Specify device
python app.py --token YOUR_HF_TOKEN --device "cuda:0"
```

3. Open the browser and go to `http://localhost:7860/`.
4. Enjoy

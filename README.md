# Latent Consistency Models
Generate images based on text prompts. LCM can generate high-quality images in very short inference time.

Inspired by [Generate images in one second on your Mac using a latent consistency model](https://replicate.com/blog/run-latent-consistency-model-on-mac) and using [Latent Consistency Models](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)

## Install
```sh
python3 -m pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
source .venv/bin/activate

# For help
python main.py --help

python main.py \
  -p "close view of a big orange-grey country house with a broken walls and mold, brush stroke style, 8k" \
  -np "tree, green"
```

## Model Download
The model will be downloaded to `~/.cache/huggingface`.
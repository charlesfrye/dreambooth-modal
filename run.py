import os
from pathlib import Path
import subprocess

import modal

stub = modal.Stub(name="stable-owl")

requirements = [
    "diffusers>==0.5.0",
    "accelerate",
    "torchvision",
    "transformers>=4.21.0",
    "ftfy",
    "tensorboard",
    "modelcards",
    "smart_open",
    "requests",
    "climage",
    "wandb",
]

image = modal.Image.debian_slim().pip_install(requirements).apt_install(["wget"])

volume = modal.SharedVolume().persist("stable-owl-training-vol")

gpu = True if os.environ.get("MODAL_GPU") else False
gpu = modal.gpu.A100() if os.environ.get("A100") else gpu
MODEL_DIR = Path("/models/stable-owl-v0")

IMG_PATH = Path(__file__).parent / "img"


def load_images(path="image_urls.txt"):
    from smart_open import open

    with open(path) as f:
        lines = f.readlines()

    image_urls = [line.strip() for line in lines]
    images = [get_image_from_url(url) for url in image_urls]

    return images


def save_images(images):
    os.makedirs(IMG_PATH, exist_ok=True)
    for ii, image in enumerate(images):
        image.save(IMG_PATH / f"{ii}.png")


def get_image_from_url(image_url):
    import io

    import requests
    import PIL.Image

    r = requests.get(image_url)
    if r.status_code != 200:
        raise RuntimeError(f"Could not download '{image_url}'")

    image = PIL.Image.open(io.BytesIO(r.content))

    return image


@stub.function(image=image, gpu=gpu)
def display_gpu_info():
    try:
        print(subprocess.run("nvidia-smi"))
    except Exception:
        print("no gpu")


@stub.function(image=image, gpu=gpu, cpu=8, shared_volumes={str(MODEL_DIR): volume}, timeout=1800, interactive=True, secret=modal.Secret.from_name("huggingface"),)
def train():
    from accelerate.utils import write_basic_config
    import huggingface_hub

    write_basic_config(mixed_precision="fp16")

    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)

    save_images(load_images("https://pastebin.com/raw/apccxadq"))

    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    INSTANCE_DIR = IMG_PATH
    OUTPUT_DIR = MODEL_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    script_url = "https://raw.githubusercontent.com/huggingface/diffusers/30220905c4319e46e114cf7dc8047d94eca226f7/examples/dreambooth/train_dreambooth.py"
    subprocess.run(
        ["wget", script_url, "-O", "train_dreambooth.py"]
    )

    subprocess.run(
        ["accelerate", "launch", "train_dreambooth.py",
        f"--pretrained_model_name_or_path={MODEL_NAME}",
        f"--instance_data_dir={INSTANCE_DIR}",
        f"--output_dir={OUTPUT_DIR}",
        "--instance_prompt='a drawing of green Duolingo owl on blank background'",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--learning_rate=5e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=400",
    ]
    )


@stub.function(image=image, gpu=gpu, cpu=1, shared_volumes={str(MODEL_DIR): volume}, interactive=True, secret=modal.Secret.from_name("wandb"))
def infer():
    from diffusers import StableDiffusionPipeline
    import torch
    import wandb

    model_id = MODEL_DIR
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    prompt = "A drawing of green Duolingo owl at a construction site next to a big crane"

    wandb.init(project="stable-owl")
    for _ in range(16):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        wandb.log({"generation": wandb.Image(image)})
    wandb.finish()
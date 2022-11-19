import os
from pathlib import Path
import subprocess

from dataclasses import dataclass, asdict
import dotenv
import modal


# load the .env file into the local environment
#  change the entries there to change the project
dotenv.load_dotenv()

@dataclass
class ProjectConfig:
    """Project-level configuration information, provided locally."""
    PROJECT_NAME: str = os.environ.get("PROJECT_NAME")
    IMAGES_FILE_URL: str = os.environ.get("IMAGES_FILE_URL")
    INSTANCE_PREFIX: str = os.environ.get("INSTANCE_PREFIX")
    INSTANCE_PHRASE: str = os.environ.get("INSTANCE_PHRASE")

project_config = ProjectConfig()
stub = modal.Stub(name=project_config.PROJECT_NAME)
stub["local_config"] = modal.Secret(asdict(project_config))

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
    "python-dotenv",
]

image = modal.Image.debian_slim().pip_install(requirements).apt_install(["wget"])
volume = modal.SharedVolume().persist(f"{project_config.PROJECT_NAME}-training-vol")

gpu = True if os.environ.get("MODAL_GPU") else False
gpu = modal.gpu.A100() if os.environ.get("A100") else gpu

MODEL_DIR = Path(f"/model")
IMG_PATH = Path(__file__).parent / "img"


@dataclass
class TrainConfig:
    """Hyperparameters/constants from the huggingface training repo."""
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 400


@stub.function(
    image=image,
    gpu=gpu,
    cpu=8,
    shared_volumes={str(MODEL_DIR): volume},
    timeout=480,
    interactive=True,
    secrets=[modal.Secret.from_name("huggingface"), stub["local_config"]],
    )
def train(config = TrainConfig()):
    from accelerate.utils import write_basic_config
    import huggingface_hub

    write_basic_config(mixed_precision="fp16")

    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)

    save_images(load_images(os.environ["IMAGES_FILE_URL"]))

    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    INSTANCE_DIR, OUTPUT_DIR = IMG_PATH, MODEL_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_repo_url = "https://raw.githubusercontent.com/huggingface/diffusers"
    script_commit_hash = "30220905c4319e46e114cf7dc8047d94eca226f7"
    script_path = "examples/dreambooth/train_dreambooth.py"
    script_url = f"{raw_repo_url}/{script_commit_hash}/{script_path}"

    subprocess.run(
        ["wget", script_url, "-O", "train_dreambooth.py"]
    )

    instance_prefix, instance_postfix = os.environ["instance_prefix"], os.environ["instance_postfix"]
    instance_phrase = os.environ["instance_phrase"]
    prompt = f"{instance_prefix} {instance_phrase} {instance_postfix}"

    subprocess.run(
        ["accelerate", "launch", "train_dreambooth.py",
        f"--pretrained_model_name_or_path={MODEL_NAME}",
        f"--instance_data_dir={INSTANCE_DIR}",
        f"--output_dir={OUTPUT_DIR}",
        f"--instance_prompt='{prompt}'",
        f"--resolution={config.resolution}",
        f"--train_batch_size={config.train_batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--learning_rate={config.learning_rate}",
        f"--lr_scheduler={config.lr_scheduler}",
        f"--lr_warmup_steps={config.lr_warmup_steps}",
        f"--max_train_steps={config.max_train_steps}",
    ]
    )


@dataclass
class InferenceConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

inference_prompts = {
    "PROMPT_POSTFIX": os.environ.get("PROMPT_POSTFIX", ""),
    "DIRECT_PROMPT": os.environ.get("DIRECT_PROMPT", "")
}
stub["inference_prompts"] = modal.Secret(inference_prompts)


@stub.function(
    image=image,
    gpu=gpu,
    cpu=1,
    shared_volumes={str(MODEL_DIR): volume},
    timeout=120,
    secrets=[modal.Secret.from_name("wandb"), stub["inference_prompts"], stub["local_config"]]
    )
def infer(config = InferenceConfig()):
    from diffusers import StableDiffusionPipeline
    import torch
    import wandb

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda")

    # create prompt based on dreambooth training instance info
    instance_prefix = os.environ["INSTANCE_PREFIX"]
    instance_phrase = os.environ["INSTANCE_PHRASE"]
    prompt_postfix = os.environ.get("PROMPT_POSTFIX", "")
    prompt = f"{instance_prefix} {instance_phrase} {prompt_postfix}"

    # or over-ride with user-supplied prompt
    prompt = os.environ.get("DIRECT_PROMPT") or prompt

    num_inference_steps = config.num_inference_steps
    guidance_scale = config.guidance_scale

    wandb.init(project=f"{os.environ['PROJECT_NAME']}", config={"prompt": prompt})
    for _ in range(16):
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        wandb.log({"generation": wandb.Image(image, caption=prompt)})
    wandb.finish()


def load_images(path):
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
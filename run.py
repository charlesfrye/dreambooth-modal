import os
from pathlib import Path
import subprocess

from dataclasses import dataclass, asdict
import dotenv
from fastapi import FastAPI
import modal


# load the .env file into the local environment
#  change the entries there to change the behavior
dotenv.load_dotenv(".env")


@dataclass
class ProjectConfig:
    """Project-level configuration information, provided locally."""

    # name of project on Modal and W&B
    PROJECT_NAME: str = os.environ.get("PROJECT_NAME")

    # url of plaintext file with urls for images of target instance
    IMAGES_FILE_URL: str = os.environ.get("IMAGES_FILE_URL")

    # training prompt looks like `{PREFIX} {PHRASE} {POSTFIX}`
    INSTANCE_PREFIX: str = os.environ.get("INSTANCE_PREFIX")
    INSTANCE_PHRASE: str = os.environ.get("INSTANCE_PHRASE")
    INSTANCE_POSTFIX: str = os.environ.get("INSTANCE_POSTFIX")


project_config = ProjectConfig()

# create an application "Stub" to coordinate local and remote execution
stub = modal.Stub(name=project_config.PROJECT_NAME)
stub["local_config"] = modal.Secret(asdict(project_config))

# list of pip-installable dependencies
requirements = [
    "diffusers>==0.5.0",
    "accelerate",
    "torchvision",
    "gradio~=3.6",
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

# from a base (container) image, add our Python and system libraries
image = modal.Image.debian_slim().pip_install(requirements).apt_install(["wget"])
# create a persistent volume to store model weights and share between components
volume = modal.SharedVolume().persist(f"{project_config.PROJECT_NAME}-training-vol")
MODEL_DIR = Path(f"/model")

# attach a server-grade GPU where needed
gpu = True if os.environ.get("MODAL_GPU") else False
gpu = modal.gpu.A100() if os.environ.get("A100") else gpu

# set a path for saving instance images
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
    gpu=gpu,  # training requires a lot of VRAM, so this should be an A100
    cpu=8,  # request enough CPUs to feed the GPU
    shared_volumes={
        str(MODEL_DIR): volume
    },  # mounts the shared volume for storing model weights
    timeout=480,
    # project-level configuration info is sent via a Secret, as are API keys
    secrets=[modal.Secret.from_name("huggingface"), stub["local_config"]],
    # interactive setups allow for easier debugging, deactivate if you hit bugs
    interactive=True,
)
def train(config=TrainConfig()):
    """Finetunes a Stable Diffusion model on a target instance.

    Run on Modal via the command line
    ```bash
        A100=1 MODAL_GPU=1 modal app run run.py --function-name train
    ```

    Adjust training details by editing the `TrainConfig`.

    Change the target instance info by editing the `.env` file.
    """
    from accelerate.utils import write_basic_config
    import huggingface_hub

    # set up local image and remote model weight directories
    save_images(load_images(os.environ["IMAGES_FILE_URL"]))
    INSTANCE_DIR, OUTPUT_DIR = IMG_PATH, MODEL_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="fp16")

    # authenticate to hugging face so we can download the model weights
    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    huggingface_hub.login(hf_key)
    MODEL_NAME = "CompVis/stable-diffusion-v1-4"

    # fetch the training script from GitHub
    raw_repo_url = "https://raw.githubusercontent.com/huggingface/diffusers"
    script_commit_hash = "30220905c4319e46e114cf7dc8047d94eca226f7"
    script_path = "examples/dreambooth/train_dreambooth.py"
    script_url = f"{raw_repo_url}/{script_commit_hash}/{script_path}"

    subprocess.run(["wget", script_url, "-O", "train_dreambooth.py"])

    # define the prompt for this instance
    instance_prefix, instance_postfix = map(
        os.environ.get, ("INSTANCE_PREFIX", "INSTANCE_POSTFIX")
    )
    instance_phrase = os.environ["INSTANCE_PHRASE"]
    prompt = f"{instance_prefix} {instance_phrase} {instance_postfix}".strip()

    # run training -- see huggingface accelerate docs for details
    subprocess.run(
        [
            "accelerate",
            "launch",
            "train_dreambooth.py",
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
    """Configuration information for inference."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5


# package up local information about inference prompt to share with Modal
inference_prompts = {
    "PROMPT_PREFIX": os.environ.get("PROMPT_PREFIX", ""),
    "PROMPT_POSTFIX": os.environ.get("PROMPT_POSTFIX", ""),
    "DIRECT_PROMPT": os.environ.get("DIRECT_PROMPT", ""),
}
stub["inference_prompts"] = modal.Secret(inference_prompts)


@stub.function(
    image=image,
    gpu=gpu,
    cpu=1,  # during inference, CPU is less of a bottleneck
    shared_volumes={str(MODEL_DIR): volume},
    timeout=120,
    secrets=[
        modal.Secret.from_name("wandb"),
        stub["inference_prompts"],
        stub["local_config"],
    ],
)
def infer(config=InferenceConfig()):
    """Run inference on Modal with a finetuned model.

    Provide prompt info via the command line, like
    ```bash
        PROMPT_PREFIX="a painting of" PROMPT_POSTFIX="in the style of Van Gogh" A100=1 MODAL_GPU=1 modal app run run.py --function-name infer
    ```
    """
    from diffusers import StableDiffusionPipeline
    import torch
    import wandb

    # set up a hugging face inference pipeline using our model
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16
    ).to("cuda")

    # create prompt based on dreambooth training instance info
    prompt_prefix = os.environ.get("PROMPT_PREFIX", "")
    instance_phrase = os.environ["INSTANCE_PHRASE"]
    prompt_postfix = os.environ.get("PROMPT_POSTFIX", "")
    prompt = f"{prompt_prefix} {instance_phrase} {prompt_postfix}"

    # or over-ride with user-supplied prompt
    prompt = os.environ.get("DIRECT_PROMPT") or prompt

    # consume inference configuration info
    num_inferences = config.num_inferences
    num_inference_steps = config.num_inference_steps
    guidance_scale = config.guidance_scale

    # create a wandb Run to send our inferences to
    wandb.init(project=f"{os.environ['PROJECT_NAME']}", config={"prompt": prompt})

    # run inference
    for _ in range(num_inferences):
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        wandb.log({"generation": wandb.Image(image, caption=prompt)})

    # close out wandb Run
    wandb.finish()


web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"


@stub.asgi(
    image=image,
    gpu=gpu,
    cpu=1,  # during inference, CPU is less of a bottleneck
    shared_volumes={str(MODEL_DIR): volume},
    mounts=[modal.Mount("/assets", local_dir=assets_path)],
)
def app(config=InferenceConfig()):
    from diffusers import StableDiffusionPipeline
    import gradio as gr
    from gradio.routes import mount_gradio_app
    import torch

    # set up a hugging face inference pipeline using our model
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16
    ).to("cuda")

    # consume inference config
    num_inference_steps = config.num_inference_steps
    guidance_scale = config.guidance_scale

    # wrap inference in a text-to-image function
    def go(text):
        image = pipe(
            text,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image

    # add a gradio UI around inference
    interface = gr.Interface(
        fn=go,
        inputs="text",
        outputs=gr.Image(shape=(512, 512)),
        css="/assets/index.css",
    )

    # mount for execution on Modal
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )


# utilities for handling images


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

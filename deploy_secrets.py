import os

import dotenv
import modal

dotenv.load_dotenv(".env.secrets")

local_wandb_secret = os.environ["WANDB_API_KEY"]
wandb_secret = modal.Stub("wandb")
wandb_secret["secret"] = modal.Secret({"WANDB_API_KEY": local_wandb_secret})

local_hf_secret = os.environ["HUGGINGFACE_TOKEN"]
hf_secret = modal.Stub("huggingface")
hf_secret["secret"] = modal.Secret({"HUGGINGFACE_TOKEN": local_hf_secret})

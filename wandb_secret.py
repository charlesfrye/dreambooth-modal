import os

import dotenv
import modal

dotenv.load_dotenv(".env.secrets")

local_wandb_secret = os.environ["WANDB_API_KEY"]
stub = modal.Stub("wandb")
stub["secret"] = modal.Secret({"WANDB_API_KEY": local_wandb_secret})

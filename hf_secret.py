import os

import dotenv
import modal

dotenv.load_dotenv(".env.secrets")

local_hf_secret = os.environ["HUGGINGFACE_TOKEN"]
stub = modal.Stub("huggingface")
stub["secret"] = modal.Secret({"HUGGINGFACE_TOKEN": local_hf_secret})

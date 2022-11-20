PHONY: help
.DEFAULT_GOAL := help

help: ## get a list of all the targets, and their short descriptions
	@# source for the incantation: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'


model: environment hf_secret ## on modal, finetune a dreambooth model to generate a target instance
	A100=1 MODAL_GPU=1 modal app run run.py --function-name train


inference: environment wandb_secret ## run inference on modal, prompting with target instance, flanked by a prefix and postfix
	PROMPT_PREFIX="$(PROMPT_PREFIX)" PROMPT_POSTFIX="$(PROMPT_POSTFIX)" A100=1 MODAL_GPU=1 modal app run run.py --function-name infer


secrets: wandb_secret hf_secret ## deploys local secrets from .env.secrets onto modal


wandb_secret: environment ## deploys wandb api key secret from .env.secrets onto modal
ifeq (,$(wildcard .env.secrets))
    $(error create a .env.secrets file with wandb api key first)
endif
	modal app deploy deploy_secrets.py::wandb_secret


hf_secret: environment ## deploys huggingface token secret from .env.secrets onto modal
ifeq (,$(wildcard .env.secrets))
    $(error create a .env.secrets file with huggingface token first)
endif
	modal app deploy deploy_secrets.py::hf_secret


environment: ## install local requirements
	pip install -r requirements-local.txt


pyfmt: dev-environment ## run formatting
	pre-commit run --all-files black


dev-environment: environment ## install local dev requirements
	pip install pre-commit
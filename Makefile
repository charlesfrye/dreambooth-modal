PHONY: help
.DEFAULT_GOAL := help

help: ## get a list of all the targets, and their short descriptions
	@# source for the incantation: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'


train: environment ## train a model on modal
	A100=1 MODAL_GPU=1 modal app run run.py --function-name train


inference: environment ## run inference on modal
	A100=1 MODAL_GPU=1 modal app run run.py --function-name infer


environment: ## install local requirements
	pip install -r requirements-local.txt
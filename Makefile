export RUNTIME_TAG := $(shell git rev-list -1 HEAD -- Dockerfile | cut -c 1-8)
export IMAGE_TAG ?= $(shell git rev-parse --abbrev-ref HEAD)-$(shell git rev-list -1 HEAD -- | cut -c 1-8)
export IMAGE ?= # TODO Registry url
export RUNTIME_IMAGE := # TODO Registry runtime url

export MODEL_STORE := /model-server/model-store


RSHELL_RUN := docker-compose run --rm -T --user=$(UID):$(GID) app
ifdef CI
	RSHELL_RUN := docker run -t --rm -e MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD}" -e RUN_ENVIRONMENT="CI" -e CI_PIPELINE_ID="${CI_PIPELINE_ID}" -e CI_COMMIT_SHA="${CI_COMMIT_SHA}" -e CI_PROJECT_URL="${CI_PROJECT_URL}" -e GITLAB_USER_NAME="${GITLAB_USER_NAME}" --volumes-from "$(shell /usr/local/bin/host-container-id)" -w "${CI_PROJECT_DIR}" "${RUNTIME_IMAGE}:${RUNTIME_TAG}"
	TORCHSERVE_RUN := docker run -d --rm -e MLFLOW_TRACKING_PASSWORD="${MLFLOW_TRACKING_PASSWORD}" -e RUN_ENVIRONMENT="CI" --volumes-from "$(shell /usr/local/bin/host-container-id)" -w "${CI_PROJECT_DIR}" "${IMAGE}:${IMAGE_TAG}"
	EXEC_IN_TORCHSERVE := docker exec torchserve-container
endif


.PHONY: train-model
train-model: #
	$(MAKE) prepare_dataset create-model-store run-model-train move-model-to-model-store

.PHONY: prepare_dataset
prepare_dataset: #
	$(RSHELL_RUN) sh -c "ls -la src/ && python src/fetch_and_prepare_raw_data.py"

.PHONY: create-model-store
create-model-store: #
	$(RSHELL_RUN) sh -c "mkdir -p ./model-server/model-store"

.PHONY: run-model-train
run-model-train: #
	$(RSHELL_RUN) sh -c "python src/main.py"

.PHONY: generate-mar-file
generate-mar-file: #
	$(RSHELL_RUN) sh -c "echo 'Creating .mar file' && torch-model-archiver --model-name model --version 1 --model-file src/modules/model.py --serialized-file ./model.pth --handler src/handler.py --export-path ./model-server/model-store && echo '.mar file created' "

.PHONY: move-model-to-model-store
move-model-to-model-store: #
	$(RSHELL_RUN) sh -c "mv ./model.mar ./model-server/model-store/"

.PHONY: runtime-build
runtime-build: ## build docker image from dockerfiles
	@docker build --cache-from "${RUNTIME_IMAGE}:${RUNTIME_TAG}" --cache-from "${RUNTIME_IMAGE}:latest" -t "${RUNTIME_IMAGE}:${RUNTIME_TAG}" -t "${RUNTIME_IMAGE}:latest" --target dev .

.PHONY: runtime-pull
runtime-pull: ## download runtime docker image from registry
	@docker pull "${RUNTIME_IMAGE}:${RUNTIME_TAG}" || true
	@docker pull "${RUNTIME_IMAGE}:latest" || true

.PHONY: runtime-remove
runtime-remove:
	@docker image remove "${RUNTIME_IMAGE}:${RUNTIME_TAG}"
	@docker image remove "${RUNTIME_IMAGE}:latest" || true

.PHONY: runtime-push
runtime-push: ## push runtime docker image to registry
	@docker push "${RUNTIME_IMAGE}:${RUNTIME_TAG}"
	@docker push "${RUNTIME_IMAGE}:latest"

.PHONY: torchserve-pull
torchserve-pull: ## download build docker image from registry
	@docker pull "${IMAGE}:${IMAGE_TAG}" || true


.PHONY: start-torchserve
start-torchserve:
	@echo "Starting TorchServe in Docker..."
	@$(TORCHSERVE_RUN) > container_id.txt

.PHONY: torchserve-test
torchserve-test: ## download runtime docker image from registry
	@$(EXEC_IN_TORCHSERVE) sh -c "python src/test/torch_serve_test.py"

.PHONY: torchserve-install
torchserve-install: ## download runtime docker image from registry
	@$(EXEC_IN_TORCHSERVE) sh -c "pip install requests"

.PHONY: stop-torchserve
stop-torchserve:
	@echo "Stopping TorchServe..."
	@docker stop $$(cat container_id.txt)
	@rm container_id.txt


.PHONY: build
build: ## build docker images from dockerfiles
	$(MAKE) runtime-build

.PHONY: docker-build
docker-build: ## build final docker image from dockerfile
	@docker build --cache-from "${RUNTIME_IMAGE}:${RUNTIME_TAG}" -t "${IMAGE}:${IMAGE_TAG}" --target prod .

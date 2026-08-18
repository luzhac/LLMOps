SHELL := /bin/bash

PYTHON ?= python3
TERRAFORM_DIR := infra/terraform
CHART := platform/helm/trade-balance-llm
NAMESPACE ?= llm-platform
RELEASE ?= trade-balance-llm

.PHONY: test lint terraform-check helm-check install upgrade model-up model-down status port-forward benchmark destroy

test:
	$(PYTHON) -m pytest app/gateway/tests -q

lint:
	$(PYTHON) -m ruff check app loadtest

terraform-check:
	terraform -chdir=$(TERRAFORM_DIR) fmt -check -recursive
	terraform -chdir=$(TERRAFORM_DIR) init -backend=false
	terraform -chdir=$(TERRAFORM_DIR) validate

helm-check:
	helm lint $(CHART)
	helm template $(RELEASE) $(CHART) --namespace $(NAMESPACE) >/dev/null

install:
	helm upgrade --install $(RELEASE) $(CHART) --namespace $(NAMESPACE) --create-namespace

upgrade: install

model-up:
	kubectl -n $(NAMESPACE) scale deployment/vllm --replicas=1
	kubectl -n $(NAMESPACE) rollout status deployment/vllm --timeout=30m

model-down:
	kubectl -n $(NAMESPACE) scale deployment/vllm --replicas=0

status:
	kubectl -n $(NAMESPACE) get pods,svc
	kubectl get nodes -L cloud.google.com/gke-accelerator,cloud.google.com/gke-spot

port-forward:
	kubectl -n $(NAMESPACE) port-forward svc/api-gateway 8080:8080

benchmark:
	$(PYTHON) loadtest/benchmark.py --base-url http://127.0.0.1:8080 --model Qwen/Qwen3-8B-AWQ

destroy:
	@echo "Use the explicit, reviewed commands in docs/runbook.md."
	@false


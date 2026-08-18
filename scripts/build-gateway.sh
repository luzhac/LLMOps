#!/usr/bin/env bash
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:=europe-west4}"
: "${IMAGE_TAG:=0.1.0}"

image="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/trade-balance-llm/gateway:${IMAGE_TAG}"
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet
docker build --tag "${image}" app/gateway
docker push "${image}"
echo "${image}"


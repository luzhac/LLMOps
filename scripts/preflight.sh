#!/usr/bin/env bash
set -euo pipefail

required=(gcloud gke-gcloud-auth-plugin terraform kubectl helm docker python3 openssl)
for command_name in "${required[@]}"; do
  command -v "${command_name}" >/dev/null || {
    echo "missing required command: ${command_name}" >&2
    exit 1
  }
done

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to the immutable project ID}"
: "${GCP_ZONE:=europe-west4-a}"

gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud billing projects describe "${GCP_PROJECT_ID}"
gcloud compute accelerator-types describe nvidia-l4 --zone "${GCP_ZONE}" --project "${GCP_PROJECT_ID}"
gcloud compute project-info describe \
  --project "${GCP_PROJECT_ID}" \
  --flatten='quotas[]' \
  --format='table(quotas.metric,quotas.limit,quotas.usage)'

echo "Preflight completed. Also confirm the billing account says Paid, not Free trial."


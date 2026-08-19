# Partial backend configuration — deliberately has no bucket name here.
#
# The bucket name contains the real GCP project ID, which this repo keeps out
# of version control. Supplying it at init time instead lets this file be
# committed safely, and (critically) lets CI find the SAME remote state as
# local runs.
#
# Why this matters: when this file was gitignored entirely, CI checked out a
# tree with no backend config at all, silently fell back to an empty LOCAL
# state, concluded that nothing existed, and tried to re-create the whole
# platform. Most resources failed with "409 already exists" (which is what
# prevented real damage) but it did create a duplicate GCS bucket.
#
# Local:
#   terraform init -backend-config=backend.hcl
# CI (see .github/workflows/deploy.yml):
#   terraform init -backend-config="bucket=$TF_STATE_BUCKET" \
#                  -backend-config="prefix=trade-balance-llm/production"
terraform {
  backend "gcs" {}
}

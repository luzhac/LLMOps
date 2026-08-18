# Bootstrap creates the resources that CI needs to already exist before it can
# authenticate and run anything itself — a classic chicken-and-egg problem.
# Run this once, locally, with your own gcloud identity:
#
#   cd infra/bootstrap
#   terraform init
#   terraform apply
#
# Never run this from CI. Its own state stays local (see versions.tf) and is
# small enough that losing it just means re-creating a few IAM/bucket
# resources, not re-provisioning the whole platform.

locals {
  required_apis = toset([
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "storage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])
}

resource "google_project_service" "bootstrap_apis" {
  for_each = local.required_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# --- Remote state bucket, shared by every environment (dev, production, ...) ---

resource "google_storage_bucket" "terraform_state" {
  project                     = var.project_id
  name                        = var.state_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  labels                      = var.labels

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.bootstrap_apis]
}

# --- Workload Identity Federation: lets GitHub Actions authenticate without a
#     stored service-account key ---

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"

  depends_on = [google_project_service.bootstrap_apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub"

  # Only tokens minted for workflow runs in this exact repo are accepted.
  attribute_condition = "assertion.repository == \"${var.github_repo}\""

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# --- CI service account and the roles it needs to run infra/environments/* ---
#
# Scoped for a single-maintainer portfolio project, not a multi-team org. A
# real company would split this into narrower per-resource-type roles and
# probably per-environment service accounts instead of one that can touch
# everything this project uses.

resource "google_service_account" "github_actions" {
  project      = var.project_id
  account_id   = "github-actions-ci"
  display_name = "GitHub Actions CI/CD"
}

locals {
  ci_roles = toset([
    "roles/container.admin",
    "roles/compute.networkAdmin",
    "roles/artifactregistry.admin",
    "roles/storage.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",
    "roles/resourcemanager.projectIamAdmin",
    "roles/serviceusage.serviceUsageAdmin",
    "roles/billing.projectManager",
  ])
}

resource "google_project_iam_member" "github_actions" {
  for_each = local.ci_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

# Let workflow runs from the configured GitHub repo impersonate this service
# account, but nothing else.
resource "google_service_account_iam_member" "github_actions_wif" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

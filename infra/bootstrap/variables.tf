variable "project_id" {
  description = "Immutable GCP project ID."
  type        = string
}

variable "region" {
  description = "Region for the Terraform state bucket. Deliberately separate from deployment_region — a multi-region bucket location like \"EU\" is not a valid compute region."
  type        = string
  default     = "EU"
}

variable "deployment_region" {
  description = "The actual GCP region environments deploy into (e.g. europe-central2). Only used here to print the GCP_REGION value for GitHub Actions — bootstrap itself creates no resources in it."
  type        = string
  default     = "europe-central2"
}

variable "github_repo" {
  description = "GitHub repo allowed to impersonate the CI service account, as \"owner/name\"."
  type        = string
  default     = "luzhac/LLMOps"
}

variable "state_bucket_name" {
  description = "Globally unique GCS bucket name for Terraform remote state."
  type        = string
}

variable "labels" {
  type = map(string)
  default = {
    application = "trade-balance-llm"
    managed-by  = "terraform-bootstrap"
  }
}

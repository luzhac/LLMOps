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
  description = "The actual GCP region environments deploy into. Only used here to print the GCP_REGION value for GitHub Actions — bootstrap itself creates no resources in it."
  type        = string
  default     = "asia-southeast1"
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

variable "cluster_name" {
  description = "Only used to name the billing budget so it is recognisable in the console."
  type        = string
  default     = "trade-balance-llm"
}

variable "billing_account_id" {
  description = "Billing account ID used only for budget creation. Leave null to skip the budget resource."
  type        = string
  default     = null
  nullable    = true
}

variable "budget_amount" {
  description = "Alerting threshold in the billing account's own currency; not a hard spending cap."
  type        = number
  default     = 30

  validation {
    condition     = var.budget_amount > 0
    error_message = "budget_amount must be greater than zero."
  }
}

variable "labels" {
  type = map(string)
  default = {
    application = "trade-balance-llm"
    managed-by  = "terraform-bootstrap"
  }
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-southeast1"
}

variable "zone" {
  type    = string
  default = "asia-southeast1-a"
}

variable "admin_cidr" {
  type = string
}

variable "artifact_bucket_location" {
  type    = string
  default = "ASIA"
}

# The billing budget moved to infra/bootstrap so CI never needs billing-account
# permissions. TF_VAR_billing_account_id in CI is now harmless but unused.
